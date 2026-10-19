import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API proxy means the browser only ever talks to the Vite origin, so the
// backend needs no CORS middleware in development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
