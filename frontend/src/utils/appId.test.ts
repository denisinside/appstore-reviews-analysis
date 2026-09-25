import { describe, expect, it } from "vitest";
import { parseAppStoreId } from "./appId";

describe("parseAppStoreId", () => {
  it.each([
    ["512939461", "512939461"],
    [" 512939461 ", "512939461"],
    ["id512939461", "512939461"],
    ["ID512939461", "512939461"],
    ["https://apps.apple.com/ru/app/subway-surfers/id512939461", "512939461"],
    ["https://apps.apple.com/us/app/example/id512939461?l=ru", "512939461"],
  ])("parses %s", (input, expected) => {
    expect(parseAppStoreId(input)).toBe(expected);
  });

  it.each([
    "",
    "5129abc",
    "0",
    "id",
    "id000",
    "https://example.com/app/id512939461",
    "http://apple.com/app/id512939461",
    "apps.apple.com/us/app/id512939461",
    "https://apps.apple.com/us/app/example",
    "https://apps.apple.com.evil.test/us/app/id512939461",
    "https://apps.apple.com:8443/us/app/example/id512939461",
  ])("rejects %s", (input) => {
    expect(parseAppStoreId(input)).toBeNull();
  });
});
