# Fragment Reconstruction Canvas - POC

A web-based canvas application for Buddhist manuscript fragment reconstruction. This POC focuses on drag-and-drop functionality for manipulating fragment images on an interactive canvas.

## Features

### Core Functionality
- **Drag & Drop**: Drag fragment images from the sidebar onto the canvas
- **Fragment Manipulation**:
  - Move fragments by dragging
  - Resize and rotate using transform handles
  - Select single fragments with click
  - Multi-select fragments with Shift/Ctrl + click
- **Locking**:
  - Lock fragments to prevent accidental modification
  - Unlock to resume editing
  - Works with single or multiple selected fragments
- **Canvas Management**:
  - Delete selected fragments
  - Clear entire canvas
  - Reset view

## Technology Stack

- **React 19** with TypeScript
- **Vite** - Fast build tool and dev server
- **react-konva** - React wrapper for Konva.js canvas library
- **Tailwind CSS** - Utility-first styling

## Getting Started

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

The application will be available at `http://localhost:5173/`

## Project Structure

```
web-canvas/
├── src/
│   ├── components/
│   │   ├── Canvas.tsx       # Main canvas component with Konva
│   │   ├── Sidebar.tsx      # Fragment library sidebar
│   │   └── Toolbar.tsx      # Top toolbar with controls
│   ├── types/
│   │   └── fragment.ts      # TypeScript type definitions
│   ├── utils/
│   │   └── fragments.ts     # Fragment data and utilities
│   ├── App.tsx              # Main application component
│   ├── index.css            # Global styles with Tailwind
│   └── main.tsx             # Application entry point
├── public/
│   └── images/              # Fragment image assets
└── package.json
```

## Usage Guide

### Adding Fragments to Canvas
1. Browse fragment thumbnails in the left sidebar
2. Drag a fragment thumbnail onto the canvas
3. Drop it at your desired location

### Manipulating Fragments
- **Move**: Click and drag a fragment
- **Resize/Rotate**: Select a fragment to show transform handles
  - Corner handles: resize
  - Rotation handle: rotate
- **Select**: Click on a fragment
- **Multi-select**: Hold Shift or Ctrl/Cmd and click additional fragments
- **Deselect**: Click on empty canvas area

### Using the Toolbar
- **Lock**: Lock selected fragments to prevent modification
- **Unlock**: Unlock selected fragments
- **Delete**: Remove selected fragments from canvas
- **Reset View**: Deselect all fragments
- **Clear Canvas**: Remove all fragments (with confirmation)

### Locked Fragments
- Locked fragments appear with reduced opacity (70%)
- Cannot be moved, resized, or rotated
- Can still be selected and unlocked
- Useful for anchoring positioned fragments

## Future Enhancements

This POC establishes the foundation for:
- Backend integration for fragment persistence
- Advanced search and filtering
- AI-powered fragment matching suggestions
- Collaborative features
- Zoom and pan controls
- Undo/redo functionality
- Export capabilities
- Fragment metadata display

## Development

### Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint

### Adding New Fragments

To add new fragment images:
1. Place JPG files in `public/images/`
2. Update `src/utils/fragments.ts` with the new filenames

## Technical Notes

- Canvas uses Konva.js for high-performance 2D rendering
- All fragment images are loaded dynamically from the public directory
- State management is handled with React hooks (useState, useRef)
- Responsive layout with Tailwind CSS flexbox utilities
- TypeScript for type safety and better developer experience

## License

Part of the Buddhist Studies Manuscript Fragment Reconstruction Platform capstone project.
