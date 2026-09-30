import { APIProxy } from "..";

describe("APIProxy shared parameters", () => {
  beforeEach(() => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      status: 200,
      url: "http://localhost/api/annotations/10",
      headers: new Headers(),
      text: async () => "{}",
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("can omit shared parameters from an endpoint request body", async () => {
    const api = new APIProxy({
      gateway: "http://localhost/api",
      sharedParams: { project: 3 },
      endpoints: {
        updateAnnotation: {
          path: "/annotations/:annotationID",
          method: "patch",
          includeSharedParamsInBody: false,
        },
      },
    });

    await api.updateAnnotation({ annotationID: 10 }, { body: { quality_level: 3 } });

    const [url, request] = global.fetch.mock.calls[0];
    expect(url).toContain("project=3");
    expect(JSON.parse(request.body)).toEqual({ quality_level: 3 });
  });

  it("includes shared parameters in request bodies by default", async () => {
    const api = new APIProxy({
      gateway: "http://localhost/api",
      sharedParams: { project: 3 },
      endpoints: {
        update: { path: "/items/:id", method: "patch" },
      },
    });

    await api.update({ id: 10 }, { body: { value: "updated" } });

    const [, request] = global.fetch.mock.calls[0];
    expect(JSON.parse(request.body)).toEqual({ project: 3, value: "updated" });
  });
});
