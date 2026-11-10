import { useCallback, useContext, useEffect, useRef, useState } from "react";
import { generatePath, useHistory } from "react-router";
import { NavLink } from "react-router-dom";
import { Spinner } from "../../components";
import { Button } from "../../components/Button/Button";
import { modal } from "../../components/Modal/Modal";
import { Space } from "../../components/Space/Space";
import { useAPI } from "../../providers/ApiProvider";
import { useLibrary } from "../../providers/LibraryProvider";
import { useProject } from "../../providers/ProjectProvider";
import { useContextProps, useFixedLocation, useParams } from "../../providers/RoutesProvider";
import { addAction, addCrumb, deleteAction, deleteCrumb } from "../../services/breadrumbs";
import { Block, Elem } from "../../utils/bem";
import { isDefined } from "../../utils/helpers";
import { ImportModal } from "../CreateProject/Import/ImportModal";
import { ExportPage } from "../ExportPage/ExportPage";
import { APIConfig } from "./api-config";
import { ToastContext } from "../../components/Toast/Toast";
import { FF_OPTIC_2, isFF } from "../../utils/feature-flags";

import "./DataManager.styl";

const initializeDataManager = async (root, props, params) => {
  if (!window.LabelStudio) throw Error("Label Studio Frontend doesn't exist on the page");
  if (!root && root.dataset.dmInitialized) return;

  root.dataset.dmInitialized = true;

  const { ...settings } = root.dataset;

  const dmConfig = {
    root,
    projectId: params.id,
    apiGateway: `${window.APP_SETTINGS.hostname}/api/dm`,
    apiVersion: 2,
    project: params.project,
    polling: !window.APP_SETTINGS,
    showPreviews: false,
    apiEndpoints: APIConfig.endpoints,
    interfaces: {
      import: true,
      export: true,
      backButton: false,
      labelingHeader: false,
      autoAnnotation: params.autoAnnotation,
    },
    labelStudio: {
      keymap: window.APP_SETTINGS.editor_keymap,
    },
    ...props,
    ...settings,
  };

  return new window.DataManager(dmConfig);
};

const buildLink = (path, params) => {
  return generatePath(`/projects/:id${path}`, params);
};

export const DataManagerPage = ({ ...props }) => {
  const toast = useContext(ToastContext);
  const root = useRef();
  const params = useParams();
  const history = useHistory();
  const api = useAPI();
  const { project } = useProject();
  const LabelStudio = useLibrary("lsf");
  const DataManager = useLibrary("dm");
  const setContextProps = useContextProps();
  const [crashed, setCrashed] = useState(false);
  const dataManagerRef = useRef();
  const projectId = project?.id;

  const init = useCallback(async () => {
    if (!LabelStudio) return;
    if (!DataManager) return;
    if (!root.current) return;
    if (!project?.id) return;
    if (dataManagerRef.current) return;

    const mlBackends = await api.callApi("mlBackends", {
      params: { project: project.id },
    });

    const interactiveBacked = (mlBackends ?? []).find(({ is_interactive }) => is_interactive);
    const predictionBackend =
      (mlBackends ?? []).find((b) => !b.is_interactive && b.state === "CO") ||
      (mlBackends ?? []).find((b) => !b.is_interactive) ||
      interactiveBacked;

    const dataManager = (dataManagerRef.current =
      dataManagerRef.current ??
      (await initializeDataManager(root.current, props, {
        ...params,
        project,
        autoAnnotation: isDefined(interactiveBacked),
      })));

    dataManager.mlBackends = mlBackends ?? [];
    dataManager.predictionBackend = predictionBackend;
    Object.assign(window, { dataManager });

    dataManager.on("crash", () => setCrashed());

    dataManager.on("settingsClicked", () => {
      history.push(buildLink("/settings/labeling", { id: params.id }));
    });

    dataManager.on("importClicked", () => {
      history.push(buildLink("/data/import", { id: params.id }));
    });

    dataManager.on("exportClicked", () => {
      history.push(buildLink("/data/export", { id: params.id }));
    });

    dataManager.on("error", (response) => {
      api.handleError(response);
    });

    dataManager.on("toast", ({ message, type }) => {
      toast.show({ message, type });
    });

    dataManager.on("navigate", (route) => {
      const target = route.replace(/^projects/, "");

      if (target) history.push(buildLink(target, { id: params.id }));
      else history.push("/projects/");
    });

    if (interactiveBacked) {
      dataManager.on("lsf:regionFinishedDrawing", (reg, group) => {
        const { lsf, task, currentAnnotation: annotation } = dataManager.lsf;
        const ids = group.map((r) => r.cleanId);
        const result = annotation.serializeAnnotation().filter((res) => ids.includes(res.id));

        const suggestionsRequest = api.callApi("mlInteractive", {
          params: { pk: interactiveBacked.id },
          body: {
            task: task.id,
            context: { result },
          },
        });

        // we'll check that we are processing the same task
        const wrappedRequest = new Promise(async (resolve, reject) => {
          const response = await suggestionsRequest;

          // right now task might be an old task,
          // so in order to get a current one we need to get it from lsf
          if (task.id === dataManager.lsf.task.id) {
            resolve(response);
          } else {
            reject();
          }
        });

        lsf.loadSuggestions(wrappedRequest, (response) => {
          if (response.data) {
            return response.data.result;
          }

          return null;
        });
      });
    }

    setContextProps({ dmRef: dataManager });
  }, [LabelStudio, DataManager, projectId]);

  const destroyDM = useCallback(() => {
    if (dataManagerRef.current) {
      dataManagerRef.current.destroy();
      dataManagerRef.current = null;
    }
  }, [dataManagerRef]);

  useEffect(() => {
    init();

    return () => destroyDM();
  }, [root, init]);

  if (!DataManager || !LabelStudio) {
    return (
      <div
        style={{
          flex: 1,
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Spinner size={64} />
      </div>
    );
  }

  return crashed ? (
    <Block name="crash">
      <Elem name="info">Project was deleted or not yet created</Elem>

      <Button to="/projects">Back to projects</Button>
    </Block>
  ) : (
    <Block ref={root} name="datamanager" />
  );
};

DataManagerPage.path = "/data";
DataManagerPage.pages = {
  ExportPage,
  ImportModal,
};
DataManagerPage.context = ({ dmRef }) => {
  const location = useFixedLocation();
  const { project } = useProject();
  const api = useAPI();
  const toast = useContext(ToastContext);
  const [mode, setMode] = useState(dmRef?.mode ?? "explorer");
  const [predicting, setPredicting] = useState(false);

  const getRegionLabel = (r) => {
    const v = r?.value || {};
    const arr =
      v.polygonlabels ||
      v.brushlabels ||
      v.rectanglelabels ||
      v.ellipsebasedpolygonlabels ||
      v.keypointlabels ||
      v.labels ||
      [];
    return arr[0] || null;
  };

  const isModelRegion = (r) =>
    String(r?.id || "").startsWith("tasdf-") || r?.origin === "prediction" || r?.origin === "prediction-changed";

  const runSmartPrediction = useCallback(async () => {
    const lsfWrapper = dmRef?.lsf;
    const task = lsfWrapper?.task;
    if (!task?.id) {
      toast?.show?.({ message: "Hãy mở một ảnh để chạy Smart Prediction", type: "error" });
      return;
    }
    let backend = dmRef?.predictionBackend;
    if (!backend && project?.id) {
      const list = (await api.callApi("mlBackends", { params: { project: project.id } })) ?? [];
      backend = list.find((b) => !b.is_interactive && b.state === "CO") || list.find((b) => !b.is_interactive) || list[0];
      if (dmRef) dmRef.predictionBackend = backend;
    }
    if (!backend?.id) {
      toast?.show?.({ message: "Chưa có ML Backend nào được kết nối cho dự án này", type: "error" });
      return;
    }
    setPredicting(true);
    try {
      const response = await api.callApi("predictWithML", {
        params: { pk: backend.id },
        body: { task: task.id },
      });
      if (response?.errors?.length) {
        toast?.show?.({ message: response.errors.join("; "), type: "error" });
        return;
      }
      const pred = response?.data || response;
      const predResults = pred?.result || [];
      const store = lsfWrapper?.lsf;
      let annotation = lsfWrapper?.currentAnnotation || store?.annotationStore?.selected;
      if (store?.annotationStore && (!annotation || annotation.type === "prediction")) {
        annotation = store.annotationStore.createAnnotation();
        store.annotationStore.selectAnnotation(annotation.id);
      }
      if (annotation) {
        const existing = annotation.serializeAnnotation() || [];
        const manualRegions = existing.filter((r) => !isModelRegion(r));
        const predLabels = new Set(predResults.map(getRegionLabel).filter(Boolean));
        const manualOverlap = manualRegions.some((r) => predLabels.has(getRegionLabel(r)));
        let keepOverlappingManual = false;
        if (manualOverlap) {
          keepOverlappingManual = window.confirm(
            "Phát hiện vùng bạn đã vẽ trùng nhãn với dự đoán mô hình.\nNhấn OK để giữ nguyên vùng bạn đã vẽ và chỉ thêm nhãn còn thiếu, hoặc Cancel để cập nhật theo dự đoán mới của mô hình.",
          );
        }
        let merged;
        if (keepOverlappingManual) {
          const manualLabels = new Set(manualRegions.map(getRegionLabel).filter(Boolean));
          merged = [...manualRegions, ...predResults.filter((r) => !manualLabels.has(getRegionLabel(r)))];
        } else {
          merged = [...manualRegions.filter((r) => !predLabels.has(getRegionLabel(r))), ...predResults];
        }
        annotation.deserializeAnnotation(merged);
        annotation.updateObjects?.();
      }
      toast?.show?.({
        message: `Smart Prediction (${pred?.model_version || backend.title}): Đã nạp ${predResults.length} vùng đa giác cho ảnh #${task.id}`,
        type: "info",
      });
    } finally {
      setPredicting(false);
    }
  }, [dmRef, project, api, toast]);

  useEffect(() => {
    window.__runSmartPrediction = runSmartPrediction;
    return () => {
      delete window.__runSmartPrediction;
    };
  }, [runSmartPrediction]);

  const links = {
    "/settings": "Settings",
  };

  const updateCrumbs = (currentMode) => {
    const isExplorer = currentMode === "explorer";
    const dmPath = location.pathname.replace(DataManagerPage.path, "");

    if (isExplorer) {
      deleteAction(dmPath);
      deleteCrumb("dm-crumb");
    } else {
      if (!isFF(FF_OPTIC_2)) {
        addAction(dmPath, (e) => {
          e.preventDefault();
          e.stopPropagation();
          dmRef?.store?.closeLabeling?.();
        });
      }
      addCrumb({
        key: "dm-crumb",
        title: "Labeling",
      });
    }
  };

  const showLabelingInstruction = (currentMode) => {
    const isLabelStream = currentMode === "labelstream";
    const { expert_instruction, show_instruction } = project;

    if (isLabelStream && show_instruction && expert_instruction) {
      modal({
        title: "Labeling Instructions",
        body: <div dangerouslySetInnerHTML={{ __html: expert_instruction }} />,
        style: { width: 680 },
      });
    }
  };

  const onDMModeChanged = (currentMode) => {
    setMode(currentMode);
    updateCrumbs(currentMode);
    showLabelingInstruction(currentMode);
  };

  useEffect(() => {
    if (dmRef) {
      dmRef.on("modeChanged", onDMModeChanged);
    }

    return () => {
      dmRef?.off?.("modeChanged", onDMModeChanged);
    };
  }, [dmRef, project]);

  return project && project.id ? (
    <Space size="small">
      {mode !== "explorer" && (
        <Button
          size="compact"
          look="primary"
          disabled={predicting}
          onClick={runSmartPrediction}
          style={{ background: "#0284c7", borderColor: "#0369a1", color: "#fff", fontWeight: 600 }}
        >
          {predicting ? "Predicting..." : "Smart Prediction"}
        </Button>
      )}
      <Button
        size="compact"
        onClick={() => {
          window.open(`http://${window.location.hostname}:9090/dashboard?token=tasdf-secret-token`, "_blank", "noopener");
        }}
      >
        ML Dashboard
      </Button>
      {project.expert_instruction && mode !== "explorer" && (
        <Button
          size="compact"
          onClick={() => {
            modal({
              title: "Instructions",
              body: () => <div dangerouslySetInnerHTML={{ __html: project.expert_instruction }} />,
            });
          }}
        >
          Instructions
        </Button>
      )}

      {Object.entries(links).map(([path, label]) => (
        <Button key={path} tag={NavLink} size="compact" to={`/projects/${project.id}${path}`} data-external>
          {label}
        </Button>
      ))}
    </Space>
  ) : null;
};
