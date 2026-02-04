const { FusesPlugin } = require('@electron-forge/plugin-fuses');
const { FuseV1Options, FuseVersion } = require('@electron/fuses');
const path = require('path');

module.exports = {
  packagerConfig: {
    name: 'Sanskrit Cipher',
    executableName: 'sanskrit-cipher',
    asar: true,

    // Include data folder and database
    extraResource: [
      './data',
      './electron/resources'
    ],

    // Exclude the cache directory - segmented images generated on-demand
    ignore: [
      /^\/electron\/resources\/cache\//
    ],

    // App metadata
    appBundleId: 'com.sanskritcipher.app',
    appCategoryType: 'public.app-category.education',
  },

  rebuildConfig: {},

  makers: [
    {
      name: '@electron-forge/maker-squirrel',
      config: {
        name: 'SanskritCipher'
      }
    },
    {
      name: '@electron-forge/maker-zip',
      platforms: ['darwin'],
    },
    {
      name: '@electron-forge/maker-deb',
      config: {
        options: {
          maintainer: 'Sanskrit Cipher Team',
          homepage: 'https://github.com/yourusername/sanskrit-cipher'
        }
      }
    },
    {
      name: '@electron-forge/maker-rpm',
      config: {}
    }
  ],

  plugins: [
    {
      name: '@electron-forge/plugin-auto-unpack-natives',
      config: {},
    },
    {
      name: '@electron-forge/plugin-vite',
      config: {
        // Build configuration for main and preload processes
        build: [
          {
            entry: 'electron/main/main.ts',
            config: 'vite.main.config.ts',
          },
          {
            entry: 'electron/preload/preload.ts',
            config: 'vite.preload.config.ts',
          }
        ],
        renderer: [
          {
            name: 'main_window',
            config: 'vite.config.ts',
          }
        ]
      }
    },
    new FusesPlugin({
      version: FuseVersion.V1,
      [FuseV1Options.RunAsNode]: false,
      [FuseV1Options.EnableCookieEncryption]: true,
      [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
      [FuseV1Options.EnableNodeCliInspectArguments]: false,
      [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: true,
      [FuseV1Options.OnlyLoadAppFromAsar]: true,
    }),
  ]
};
