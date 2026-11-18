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
import { acceptTarget, frameTarget, loadSettings, mergeProposal, onTarget, saveSettings } from "./propagation";

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
      (mlBackends ?? []).find((b) => !b.is_interactive);

    const dataManager = (dataManagerRef.current =
      dataManagerRef.current ??
      (await initializeDataManager(root.current, props, {
        ...params,
        project,
        autoAnnotation: isDefined(interactiveBacked),
      })));

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
  const [propagating, setPropagating] = useState(false);
  const [propSettings, setPropSettings] = useState(loadSettings);

  const updatePropSettings = (patch) => {
    setPropSettings((current) => {
      const next = { ...current, ...patch };
      saveSettings(next);
      return next;
    });
  };

  // the non-interactive backend that predicts whole images (Smart Prediction and "Done image" share it)
  const findBackend = useCallback(async () => {
    let backend = dmRef?.predictionBackend;
    if (!backend && project?.id) {
      const list = (await api.callApi("mlBackends", { params: { project: project.id } })) ?? [];
      backend = list.find((b) => !b.is_interactive && b.state === "CO") || list.find((b) => !b.is_interactive);
      if (dmRef) dmRef.predictionBackend = backend;
    }
    return backend;
  }, [dmRef, project, api]);

  // replace the annotation's regions; polygon points are re-placed on the stage they belong to
  const applyResults = (annotation, results) => {
    annotation.deleteAllRegions({ deleteReadOnly: true });
    annotation.deserializeResults(results);
    for (const r of annotation.regions || []) {
      const sw = r.parent?.stageWidth;
      const sh = r.parent?.stageHeight;
      if (r.coordstype === "perc" && sw > 1 && sh > 1) {
        for (const p of [...(r.points || []), ...(r.holes || [])]) {
          p._setPos?.((sw * p.x) / 100, (sh * p.y) / 100);
        }
      }
    }
    annotation.updateObjects?.();
  };

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

  // The image the user points at is a single image: the Repeater frame on screen (view_page) and its current slice.
  const getTargetImage = (entity) => {
    const images = [...(entity?.names?.values?.() ?? [])].filter((t) => t.type === "image");
    if (!images.length) return null;
    const page = Number.parseInt(new URLSearchParams(window.location.search).get("view_page"), 10) || 1;
    const tag = images.length > 1 ? images[Math.min(page, images.length) - 1] : images[0];
    const item_index = tag.valuelist ? tag.currentImage ?? 0 : null;
    const url = tag.valuelist ? tag.parsedValueList?.[item_index] : tag.parsedValue;
    return typeof url === "string" ? { url, to_name: tag.name, item_index } : null;
  };

  const runSmartPrediction = useCallback(async () => {
    const lsfWrapper = dmRef?.lsf;
    const task = lsfWrapper?.task;
    const store = lsfWrapper?.lsf;
    const target = getTargetImage(lsfWrapper?.currentAnnotation || store?.annotationStore?.selected);
    if (!task?.id || !target) {
      toast?.show?.({ message: "Hãy mở một ảnh để chạy Smart Prediction", type: "error" });
      return;
    }
    const backend = await findBackend();
    if (!backend?.id) {
      toast?.show?.({ message: "Chưa có ML Backend dự đoán cả ảnh (non-interactive) cho dự án này", type: "error" });
      return;
    }
    setPredicting(true);
    try {
      const response = await api.callApi("predictWithML", {
        params: { pk: backend.id },
        body: { task: task.id, force: true, image: target },
      });
      if (response?.errors?.length) {
        toast?.show?.({ message: response.errors.join("; "), type: "error" });
        return;
      }
      const pred = response?.data || response;
      const where = `${target.to_name}:${target.item_index ?? 0}`;
      // backend ids repeat on every image: make them unique per image so other frames' regions survive
      const predResults = (pred?.result || []).map((r) => ({ ...r, id: `${r.id}@${where}`, origin: "prediction" }));
      const predId = pred?.id ? Number.parseInt(pred.id, 10) : null;
      let annotation = lsfWrapper?.currentAnnotation || store?.annotationStore?.selected;
      if (store?.annotationStore && (!annotation || annotation.type === "prediction")) {
        annotation = store.annotationStore.createAnnotation({
          userGenerate: true,
          ...(predId ? { parent_prediction: predId } : {}),
        });
        store.annotationStore.selectAnnotation(annotation.id);
      }
      if (annotation) {
        if (predId && lsfWrapper) {
          // only the annotation filled here derives from this prediction (lsf-sdk prepareData checks the id)
          lsfWrapper.smartParentPrediction = { annotationId: annotation.id, predictionId: predId };
        }
        const existing = annotation.serializeAnnotation() || [];
        // only the target image is replaced; regions on other frames/slices are kept as they are
        const onTarget = (r) =>
          r.to_name === target.to_name && (target.item_index === null || (r.item_index ?? 0) === target.item_index);
        const otherImages = existing.filter((r) => !onTarget(r));
        const manualRegions = existing.filter((r) => onTarget(r) && !isModelRegion(r));
        const predLabels = new Set(predResults.map(getRegionLabel).filter(Boolean));
        const manualOverlap = manualRegions.some((r) => predLabels.has(getRegionLabel(r)));
        let keepOverlappingManual = false;
        if (manualOverlap) {
          keepOverlappingManual = window.confirm(
            "Phát hiện vùng bạn đã vẽ trùng nhãn với dự đoán mô hình.\nNhấn OK để giữ nguyên vùng bạn đã vẽ và chỉ thêm nhãn còn thiếu, hoặc Cancel để cập nhật theo dự đoán mới của mô hình.",
          );
        }
        let mergedResults;
        if (keepOverlappingManual) {
          const manualLabels = new Set(manualRegions.map(getRegionLabel).filter(Boolean));
          mergedResults = [...manualRegions, ...predResults.filter((r) => !manualLabels.has(getRegionLabel(r)))];
        } else {
          mergedResults = [...manualRegions.filter((r) => !predLabels.has(getRegionLabel(r))), ...predResults];
        }
        mergedResults = [...otherImages, ...mergedResults];
        applyResults(annotation, mergedResults);
      }
      toast?.show?.({
        message: `Smart Prediction (${pred?.model_version || backend.title}): Đã nạp ${
          predResults.length
        } vùng đa giác cho task #${task.id}, ${
          target.item_index === null ? target.to_name : `${target.to_name} slice ${target.item_index + 1}`
        }`,
        type: "info",
      });
    } finally {
      setPredicting(false);
    }
  }, [dmRef, project, api, toast, findBackend]);

  // "Done image": the image on screen becomes gold, the well adapter learns it, the next frames get proposals
  const runFrameDone = useCallback(async () => {
    const lsfWrapper = dmRef?.lsf;
    const task = lsfWrapper?.task;
    const store = lsfWrapper?.lsf;
    const annotation = lsfWrapper?.currentAnnotation || store?.annotationStore?.selected;
    const target = getTargetImage(annotation);
    if (!task?.id || !target || !annotation || annotation.type === "prediction") {
      toast?.show?.({ message: "Hãy mở một ảnh trong annotation để xác nhận", type: "error" });
      return;
    }
    const current = () => annotation.serializeAnnotation() || [];
    const carriedFrom = current().filter((r) => onTarget(r, target));
    if (!carriedFrom.length) {
      toast?.show?.({ message: "Ảnh này chưa có vùng nào để học", type: "error" });
      return;
    }
    const backend = await findBackend();
    if (!backend?.id) {
      toast?.show?.({ message: "Chưa có ML Backend dự đoán cả ảnh (non-interactive) cho dự án này", type: "error" });
      return;
    }
    const { n, threshold, auto } = propSettings;
    applyResults(annotation, acceptTarget(current(), target));
    const gold = { url: target.url, results: carriedFrom };
    const call = (image, context) =>
      api.callApi("predictWithML", {
        params: { pk: backend.id },
        body: { task: task.id, force: true, image, context: { action: "propagate", threshold, ...context } },
      });
    setPropagating(true);
    try {
      if (!auto || n === 0) {
        const resp = await call(target, { carried: carriedFrom, distance: 0, gold });
        if (resp?.errors?.length) throw new Error(resp.errors.join("; "));
        toast?.show?.({ message: "Đã học từ ảnh này", type: "info" });
        return;
      }
      const images = [...(annotation.names?.values?.() ?? [])].filter((t) => t.type === "image");
      const frames = images
        .map((tag) => ({ to_name: tag.name, urls: tag.parsedValueList ?? [tag.parsedValue] }))
        .sort((a, b) => Number(a.to_name.match(/(\d+)$/)?.[1] ?? 0) - Number(b.to_name.match(/(\d+)$/)?.[1] ?? 0));
      let carried = carriedFrom;
      let filled = 0;
      let note = "";
      for (let k = 1; k <= n; k++) {
        const next = frameTarget(frames, target, k);
        if (!next) {
          note = "đã tới frame cuối";
          break;
        }
        if (mergeProposal(current(), next, []) === null) {
          note = `gặp ảnh đã gán ở ${next.to_name}`;
          break;
        }
        const resp = await call(next, { carried, distance: k, ...(k === 1 ? { gold } : {}) });
        if (resp?.errors?.length) throw new Error(resp.errors.join("; "));
        const result = (resp?.data || resp)?.result || [];
        applyResults(annotation, mergeProposal(current(), next, result));
        filled += 1;
        carried = result;
        if (resp?.propagation?.stop) {
          note = `dừng ở ${next.to_name}: ${resp.propagation.stop}`;
          break;
        }
      }
      toast?.show?.({ message: `Đã đề xuất cho ${filled} frame${note ? `, ${note}` : ""}`, type: "info" });
    } catch (error) {
      toast?.show?.({ message: `Xong ảnh thất bại: ${error.message}`, type: "error" });
    } finally {
      setPropagating(false);
    }
  }, [dmRef, api, toast, findBackend, propSettings]);

  useEffect(() => {
    const onKey = (event) => {
      if (!event.shiftKey || event.key !== "Enter" || event.repeat) return;
      const el = document.activeElement;
      if (el?.closest?.("input, textarea, select, [contenteditable=''], [contenteditable='true']")) return;
      event.preventDefault();
      runFrameDone();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [runFrameDone]);

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
      {mode !== "explorer" && (
        <span className="tasdf-propagation">
          <Button
            size="compact"
            disabled={propagating}
            onClick={runFrameDone}
            title="Xác nhận ảnh này và lan đề xuất sang các frame sau (Shift+Enter)"
          >
            {propagating ? "Đang lan..." : "Xong ảnh"}
          </Button>
          <label htmlFor="tasdf-prop-n">N</label>
          <input
            id="tasdf-prop-n"
            type="number"
            min={0}
            max={10}
            value={propSettings.n}
            onChange={(e) => updatePropSettings({ n: Math.min(Math.max(Number(e.target.value) || 0, 0), 10) })}
          />
          <label htmlFor="tasdf-prop-threshold">Ngưỡng</label>
          <input
            id="tasdf-prop-threshold"
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={propSettings.threshold}
            onChange={(e) => updatePropSettings({ threshold: Math.min(Math.max(Number(e.target.value) || 0, 0), 1) })}
          />
          <input
            id="tasdf-prop-auto"
            type="checkbox"
            checked={propSettings.auto}
            onChange={(e) => updatePropSettings({ auto: e.target.checked })}
          />
          <label htmlFor="tasdf-prop-auto">Tự lan</label>
        </span>
      )}
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
