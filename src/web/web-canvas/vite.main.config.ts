import type { ConfigEnv, UserConfig } from 'vite';
import { defineConfig } from 'vite';
import { builtinModules } from 'node:module';

const builtins = [
  'electron',
  'electron/main',
  'electron/common',
  'electron/renderer',
  ...builtinModules.flatMap((m) => [m, `node:${m}`]),
];

// https://vitejs.dev/config
export default defineConfig((env) => {
  const forgeEnv = env as ConfigEnv & {
    forgeConfigSelf: { entry: string };
    forgeConfig: { renderer: { name: string }[] };
  };
  const { root, mode, command } = env;
  const { forgeConfigSelf, forgeConfig } = forgeEnv;

  // Generate define keys for renderer windows
  const names = forgeConfig?.renderer?.filter(({ name }) => name != null).map(({ name }) => name) ?? [];
  const define: Record<string, string | undefined> = {};
  for (const name of names) {
    const NAME = name.toUpperCase().replaceAll('-', '_');
    define[`${NAME}_VITE_DEV_SERVER_URL`] = command === 'serve' ? JSON.stringify(process.env[`${NAME}_VITE_DEV_SERVER_URL`] ?? '') : undefined;
    define[`${NAME}_VITE_NAME`] = JSON.stringify(name);
  }

  const config: UserConfig = {
    root,
    mode,
    clearScreen: false,
    build: {
      emptyOutDir: false,
      outDir: '.vite/build',
      watch: command === 'serve' ? {} : null,
      minify: command === 'build',
      lib: {
        entry: forgeConfigSelf?.entry ?? 'electron/main/main.ts',
        fileName: () => '[name].cjs',
        formats: ['cjs'],
      },
      rollupOptions: {
        external: [...builtins, 'better-sqlite3'],
      },
    },
    define,
    resolve: {
      conditions: ['node'],
      mainFields: ['module', 'jsnext:main', 'jsnext'],
    },
  };
  return config;
});
