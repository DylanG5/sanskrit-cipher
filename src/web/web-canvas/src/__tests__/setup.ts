/**
 * Vitest setup – provide stubs for browser-only globals that jsdom
 * does not include (Electron IPC, IndexedDB helpers, etc.).
 */

// Stub window.electronAPI as undefined by default; individual tests
// can assign it when needed.
Object.defineProperty(window, 'electronAPI', {
  value: undefined,
  writable: true,
  configurable: true,
});
