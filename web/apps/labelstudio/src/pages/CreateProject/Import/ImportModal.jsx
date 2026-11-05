import { useCallback, useRef, useState } from "react";
import { useHistory } from "react-router";
import { Button } from "../../../components";
import { Modal } from "../../../components/Modal/Modal";
import { Space } from "../../../components/Space/Space";
import { useAPI } from "../../../providers/ApiProvider";
import { ProjectProvider, useProject } from "../../../providers/ProjectProvider";
import { useFixedLocation } from "../../../providers/RoutesProvider";
import { Elem } from "../../../utils/bem";
import { useRefresh } from "../../../utils/hooks";
import { ImportResults } from "../../DataManager/ImportResults";
import { ImportPage } from "./Import";
import { useImportPage } from "./useImportPage";

export const Inner = () => {
  const history = useHistory();
  const location = useFixedLocation();
  const modal = useRef();
  const refresh = useRefresh();
  const { project } = useProject();
  const [waiting, setWaitingStatus] = useState(false);
  const [importMode, setImportMode] = useState("tasks");
  const [resultsBusy, setResultsBusy] = useState(false);
  const api = useAPI();

  const { uploading, uploadDisabled, finishUpload, fileIds, pageProps } = useImportPage(project);

  const backToDM = useCallback(() => {
    const path = location.pathname.replace(ImportModal.path, "");
    const search = location.search;
    const pathname = `${path}${search !== "?" ? search : ""}`;

    return refresh(pathname);
  }, [location, history]);

  const onCancel = useCallback(async () => {
    setWaitingStatus(true);
    await api.callApi("deleteFileUploads", {
      params: {
        pk: project.id,
      },
      body: {
        file_upload_ids: fileIds,
      },
    });
    setWaitingStatus(false);
    modal?.current?.hide();
    backToDM();
  }, [modal, project, fileIds, backToDM]);

  const onFinish = useCallback(async () => {
    const imported = await finishUpload();

    if (!imported) return;
    backToDM();
  }, [backToDM, finishUpload]);

  return (
    <Modal
      title="Import data"
      ref={modal}
      onHide={() => backToDM()}
      closeOnClickOutside={false}
      allowClose={!waiting && !uploading && !resultsBusy}
      fullscreen
      visible
      bare
    >
      <Modal.Header divided>
        <Elem block="modal" name="title">
          Import
        </Elem>

        <Space>
          <Button
            waiting={waiting}
            disabled={uploading || resultsBusy}
            onClick={importMode === "tasks" ? onCancel : backToDM}
          >
            Cancel
          </Button>
          {importMode === "tasks" && (
            <Button look="primary" onClick={onFinish} waiting={waiting || uploading} disabled={uploadDisabled}>
              Import
            </Button>
          )}
        </Space>
      </Modal.Header>
      {project.can_import_results && (
        <div style={{ padding: "16px 24px" }}>
          <Space>
            <Button
              look={importMode === "tasks" ? "primary" : undefined}
              disabled={waiting || uploading || resultsBusy}
              onClick={() => setImportMode("tasks")}
            >
              New tasks
            </Button>
            <Button
              look={importMode === "results" ? "primary" : undefined}
              disabled={waiting || uploading || resultsBusy || fileIds.length > 0}
              onClick={() => setImportMode("results")}
            >
              Update task
            </Button>
          </Space>
          {fileIds.length > 0 && <p>Finish or cancel the current task import before updating existing tasks.</p>}
        </div>
      )}
      {importMode === "tasks" ? (
        <ImportPage project={project} {...pageProps} />
      ) : (
        <ImportResults project={project} onBusyChange={setResultsBusy} onClose={backToDM} />
      )}
    </Modal>
  );
};
export const ImportModal = () => {
  return (
    <ProjectProvider>
      <Inner />
    </ProjectProvider>
  );
};

ImportModal.path = "/import";
ImportModal.modal = true;
