/** Normalize a pasted App Store ID or Apple app URL to its numeric ID. */
export function parseAppStoreId(input: string): string | null {
  const value = input.trim();
  if (!value) return null;

  if (/^\d+$/.test(value)) return /[1-9]/.test(value) ? value : null;

  const prefixed = value.match(/^id(\d+)$/i);
  if (prefixed) return /[1-9]/.test(prefixed[1]) ? prefixed[1] : null;

  // Only accept Apple's App Store host. Require a URL scheme so malformed or
  // arbitrary host-like text cannot be mistaken for a valid App Store link.
  if (!/^https?:\/\//i.test(value)) return null;
  try {
    const url = new URL(value);
    if (url.protocol !== "https:" && url.protocol !== "http:") return null;
    if (url.hostname.toLowerCase() !== "apps.apple.com") return null;
    if (url.username || url.password || url.port) return null;
    const match = url.pathname.match(/(?:^|\/)id(\d+)(?:\/|$)/i);
    return match && /[1-9]/.test(match[1]) ? match[1] : null;
  } catch {
    return null;
  }
}
