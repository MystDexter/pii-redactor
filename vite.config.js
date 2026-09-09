import { defineConfig } from 'vite'

// Honour the PORT env var (set by the preview manager when autoPort is on),
// falling back to Vite's default. strictPort:false lets it pick the next free
// port if the assigned one is taken.
export default defineConfig({
  server: {
    port: Number(process.env.PORT) || 5173,
    strictPort: false,
  },
})
