import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, getScan } from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("API client errors", () => {
  it("converts FastAPI detail responses into ApiError", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: "Scan not found" }),
      { status: 404, headers: { "content-type": "application/json" } },
    )));

    await expect(getScan("missing")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      message: "Scan not found",
    });
  });

  it("retains validation details when FastAPI returns a detail array", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: [{ msg: "Field required" }, { msg: "Invalid value" }] }),
      { status: 422, headers: { "content-type": "application/json" } },
    )));

    try {
      await getScan("bad");
      throw new Error("expected request to reject");
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect(error).toMatchObject({ status: 422, message: "Field required; Invalid value" });
      expect((error as ApiError).detail).toHaveLength(2);
    }
  });

  it("reports network failures as status zero", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));
    await expect(getScan("abc")).rejects.toMatchObject({ status: 0, message: "offline" });
  });
});
