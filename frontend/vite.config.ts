import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsconfigPaths from "vite-tsconfig-paths";

// Local dev only: the backend serves API + built frontend from the same origin
// in production, so src/lib/api.ts deliberately uses relative paths on
// localhost (see _resolveApiUrl). Running the frontend on its own Vite port
// against a separately-running backend needs this proxy to bridge that gap —
// it doesn't affect the production build, which never goes through vite dev.
const BACKEND_URL = process.env.VITE_DEV_BACKEND_URL || "http://localhost:8501";

export default defineConfig({
  plugins: [react(), tailwindcss(), tsconfigPaths()],
  server: {
    port: process.env.PORT ? Number(process.env.PORT) : 5173,
    strictPort: !!process.env.PORT,
    proxy: {
      "/ask-stream": BACKEND_URL,
      "/auth": BACKEND_URL,
      "/admin": BACKEND_URL,
      "/conversation": BACKEND_URL,
      "/docs": BACKEND_URL,
      "/session": BACKEND_URL,
      "/upload": BACKEND_URL,
      "/videos": BACKEND_URL,
      "/webpages": BACKEND_URL,
      "/super-admin": BACKEND_URL,
      "/tunnel-url": BACKEND_URL,
      "/health": BACKEND_URL,
      "/ws": { target: BACKEND_URL, ws: true },
    },
  },
  build: {
    outDir: "dist",
  },
});
