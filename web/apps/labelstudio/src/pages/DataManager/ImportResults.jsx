import { useState } from "react";
import { Button } from "../../components/Button/Button";
import { useAPI } from "../../providers/ApiProvider";

export const ImportResults = ({ project, onClose, onBusyChange }) => {
  const api = useAPI();
  const [tasks, setTasks] = useState(null);
  const [filename, setFilename] = useState("");
  const [matchField, setMatchField] = useState("id");
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);
  const dataFields = [...new Set((tasks ?? []).flatMap((task) => Object.keys(task?.data ?? {})))];

  const readFile = async (event) => {
    const file = event.target.files?.[0];
    setTasks(null);
    setPreview(null);
    setError("");
    setMatchField("id");
    setFilename(file?.name ?? "");
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      const items = Array.isArray(parsed) ? parsed : [parsed];
      if (!items.length || items.some((item) => !item || typeof item !== "object" || Array.isArray(item))) {
        throw new Error("JSON must contain a task object or a list of task objects.");
      }
      setTasks(items);
    } catch (err) {
      setError(err.message);
    }
  };

  const submit = async (commit) => {
    setBusy(true);
    onBusyChange?.(true);
    setError("");
    try {
      const response = await api.callApi("importProjectResults", {
        params: { pk: project.id },
        body: {
          tasks,
          match_field: matchField,
          commit,
          preview_token: preview?.preview_token,
        },
      });
      if (!response?.$meta?.ok) {
        setPreview(null);
        setError("Task update failed. Check the error details and preview again.");
        return;
      }
      setPreview(response);
      setDone(response.committed);
    } catch (err) {
      setPreview(null);
      setError(err.message);
    } finally {
      setBusy(false);
      onBusyChange?.(false);
    }
  };

  const summary = preview?.summary;
  return (
    <div style={{ padding: 24, display: "grid", gap: 16 }}>
      <p>
        Update existing tasks from Label Studio task JSON. Update data and meta keys, and add annotations or predictions
        while keeping existing results.
      </p>
      {!done && (
        <>
          <label>
            JSON task file
            <input type="file" accept=".json,application/json" disabled={busy} onChange={readFile} />
          </label>
          <label>
            Match existing tasks by{" "}
            <select
              value={matchField}
              disabled={busy}
              onChange={(event) => {
                setMatchField(event.target.value);
                setPreview(null);
              }}
            >
              <option value="id">Task ID</option>
              {dataFields.map((field) => (
                <option key={field} value={`data.${field}`}>{`data.${field}`}</option>
              ))}
            </select>
          </label>
          <p>
            Use a data field when external task IDs differ. Values must match exactly and identify one task. Use Task ID
            to change the field you would otherwise match by.
          </p>
          <p>
            Only supplied data and meta keys are replaced. Other keys are kept. A supplied nested object or array
            replaces that key's entire value. Annotations and predictions are optional.
          </p>
          <p>
            Annotations are added at the Annotator stage for review. Predictions remain predictions. External authors
            and IDs are not copied. Uploading the same file again adds another set of results.
          </p>
          <Button disabled={!tasks || busy} waiting={busy} onClick={() => submit(false)}>
            Preview
          </Button>
        </>
      )}
      {error && <p role="alert">{error}</p>}
      {summary && (
        <>
          <p role="status">
            {done ? "Updated" : filename}: {summary.matched} matched tasks, {summary.unmatched} unmatched,{" "}
            {summary.errors} errors; {summary.tasks_updated} tasks with information changes; {summary.annotations}{" "}
            annotations and {summary.predictions} predictions {done ? "added" : "to add"}.
          </p>
          {!done && (
            <>
              {summary.unmatched > 0 && <p>Unmatched rows will be skipped when you confirm.</p>}
              <div style={{ maxHeight: 280, overflow: "auto" }}>
                <table style={{ width: "100%", textAlign: "left" }}>
                  <thead>
                    <tr>
                      <th>Row</th>
                      <th>Task</th>
                      <th>Status</th>
                      <th>Details</th>
                      <th>Information changes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row) => (
                      <tr key={row.row}>
                        <td>{row.row}</td>
                        <td>{row.task_id ?? "—"}</td>
                        <td>{row.status}</td>
                        <td>{row.message ?? `${row.annotations} annotations, ${row.predictions} predictions`}</td>
                        <td>
                          {Object.keys(row.changes ?? {}).length === 0 ? (
                            "No information changes"
                          ) : (
                            <table style={{ width: "100%", textAlign: "left" }}>
                              <thead>
                                <tr>
                                  <th>Field</th>
                                  <th>Before</th>
                                  <th>After</th>
                                </tr>
                              </thead>
                              <tbody>
                                {Object.entries(row.changes).flatMap(([field, keys]) =>
                                  Object.entries(keys).map(([key, change]) => (
                                    <tr key={`${field}.${key}`}>
                                      <td>{`${field}.${key}`}</td>
                                      <td>
                                        <pre
                                          style={{
                                            whiteSpace: "pre-wrap",
                                            overflowWrap: "anywhere",
                                          }}
                                        >
                                          {change.before_exists ? JSON.stringify(change.before, null, 2) : "(not set)"}
                                        </pre>
                                      </td>
                                      <td>
                                        <pre
                                          style={{
                                            whiteSpace: "pre-wrap",
                                            overflowWrap: "anywhere",
                                          }}
                                        >
                                          {JSON.stringify(change.after, null, 2)}
                                        </pre>
                                      </td>
                                    </tr>
                                  )),
                                )}
                              </tbody>
                            </table>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Button
                look="primary"
                disabled={busy || summary.errors > 0 || summary.matched === 0}
                waiting={busy}
                onClick={() => submit(true)}
              >
                Confirm: update task
              </Button>
            </>
          )}
        </>
      )}
      <Button disabled={busy} onClick={() => onClose(done)}>
        {done ? "Done" : "Cancel"}
      </Button>
    </div>
  );
};
