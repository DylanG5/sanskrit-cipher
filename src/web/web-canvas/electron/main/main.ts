import { app, BrowserWindow, protocol } from 'electron';
import path from 'node:path';
import fs from 'node:fs';
import { initDatabase, closeDatabase } from './database/connection';
import { registerIpcHandlers } from './ipc/handlers';
import squirrelStartup from 'electron-squirrel-startup';

// Add Resources/node_modules to module search paths for packaged app
if (app.isPackaged) {
  const resourcesPath = process.resourcesPath;
  const nodeModulesPath = path.join(resourcesPath, 'node_modules');
  if (fs.existsSync(nodeModulesPath)) {
    (module as any).paths.push(nodeModulesPath);
  }
}

// Handle Squirrel events (Windows installer)
if (squirrelStartup) {
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
    // In packaged app, load from unpacked directory if it exists
    let htmlPath = path.join(__dirname, `../renderer/${MAIN_WINDOW_VITE_NAME}/index.html`);

    // Check if running from ASAR and unpacked version exists
    if (htmlPath.includes('.asar')) {
      const unpackedPath = htmlPath.replace('app.asar', 'app.asar.unpacked');
      if (fs.existsSync(unpackedPath)) {
        htmlPath = unpackedPath;
        console.log('Using unpacked renderer files from:', htmlPath);
      }
    }

    // Use loadURL with file:// protocol to properly load from unpacked directory
    const fileUrl = `file://${htmlPath}`;
    console.log('Loading URL:', fileUrl);
    mainWindow.loadURL(fileUrl);
  }

  // Open DevTools to debug rendering issues
  mainWindow.webContents.openDevTools();

  // Log the actual loaded URL
  mainWindow.webContents.on('did-finish-load', () => {
    console.log('Finished loading:', mainWindow.webContents.getURL());
  });
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
        : path.join(process.resourcesPath, 'cache/segmented');
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
