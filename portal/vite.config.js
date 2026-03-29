import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

function resolveBasePath() {
  const raw = process.env.PORTAL_BASE_URL?.trim();
  if (!raw) {
    return "/portal/";
  }

  try {
    const pathname = new URL(raw).pathname;
    return pathname.endsWith("/") ? pathname : `${pathname}/`;
  } catch {
    if (raw.startsWith("/")) {
      return raw.endsWith("/") ? raw : `${raw}/`;
    }
    return `/${raw.replace(/^\/+|\/+$/g, "")}/`;
  }
}

export default defineConfig({
  plugins: [react()],
  base: resolveBasePath(),
  server: {
    host: "0.0.0.0",
    port: 5000,
    allowedHosts: ["frps.paconsultings.com"],
  },
});
