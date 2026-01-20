import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Plugin to expose renderer dev server URL to main process
function pluginExposeRenderer(name: string): Plugin {
  return {
    name: '@electron-forge/plugin-vite:expose-renderer',
    configureServer(server) {
      server.httpServer?.once('listening', () => {
        const addressInfo = server.httpServer?.address();
        if (addressInfo && typeof addressInfo === 'object') {
          process.env[`${name.toUpperCase().replaceAll('-', '_')}_VITE_DEV_SERVER_URL`] =
            `http://localhost:${addressInfo.port}`;
        }
      });
    },
  };
}

// https://vite.dev/config/
export default defineConfig((env) => {
  const forgeEnv = env as typeof env & {
    forgeConfigSelf: { name: string };
  };
  const { forgeConfigSelf } = forgeEnv;
  const name = forgeConfigSelf?.name ?? 'main_window';

  return {
    plugins: [react(), pluginExposeRenderer(name)],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    base: './',
    build: {
      outDir: `.vite/renderer/${name}`,
    },
    clearScreen: false,
  }
})
