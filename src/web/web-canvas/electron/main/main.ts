import { app, BrowserWindow, protocol } from 'electron';
import path from 'node:path';
import fs from 'node:fs';
import { initDatabase, closeDatabase } from './database/connection';
import { registerIpcHandlers } from './ipc/handlers';

// Handle Squirrel events (Windows installer)
if (require('electron-squirrel-startup')) {
  app.quit();
}

// Declare __dirname for ES module compatibility
declare const MAIN_WINDOW_VITE_DEV_SERVER_URL: string | undefined;
declare const MAIN_WINDOW_VITE_NAME: string;

const createWindow = () => {
  // Create the browser window
  const mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // Load the app
  if (MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(MAIN_WINDOW_VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(
      path.join(__dirname, `../renderer/${MAIN_WINDOW_VITE_NAME}/index.html`)
    );
  }

  // Open DevTools in development
  if (MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    mainWindow.webContents.openDevTools();
  }
};

// Register custom protocol for loading images
const registerImageProtocol = () => {
  protocol.handle('electron-image', async (request) => {
    const url = request.url.replace('electron-image://', '');
    const decodedUrl = decodeURIComponent(url);

    const isDev = !app.isPackaged;

    let imagePath: string;

    // Check if this is a segmented image request
    if (decodedUrl.startsWith('segmented/')) {
      // Segmented images are stored in resources/cache/segmented/
      const filename = decodedUrl.replace('segmented/', '');
      const basePath = isDev
        ? path.join(process.cwd(), 'electron/resources/cache/segmented')
        : path.join(process.resourcesPath, 'resources', 'cache', 'segmented');
      imagePath = path.join(basePath, filename);
    } else {
      // Original images are in the data folder
      const basePath = isDev
        ? path.join(process.cwd(), 'data')
        : path.join(process.resourcesPath, 'data');
      imagePath = path.join(basePath, decodedUrl);
    }

    try {
      const data = await fs.promises.readFile(imagePath);
      const ext = path.extname(imagePath).toLowerCase();
      const mimeType = ext === '.png' ? 'image/png' : 'image/jpeg';

      return new Response(data, {
        headers: { 'content-type': mimeType }
      });
    } catch (error) {
      console.error('Image not found:', imagePath);
      return new Response('', { status: 404 });
    }
  });
};

// App lifecycle
app.whenReady().then(() => {
  // Initialize database
  initDatabase();

  // Register IPC handlers
  registerIpcHandlers();

  // Register custom protocol for images
  registerImageProtocol();

  // Create main window
  createWindow();

  app.on('activate', () => {
    // On macOS, re-create window when dock icon is clicked
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Cleanup on quit
app.on('will-quit', () => {
  closeDatabase();
});
