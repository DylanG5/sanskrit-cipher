#!/bin/bash

# Set source and destination directories
SOURCE_DIR=~/Downloads
DEST_DIR=~/capstone_data

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Find all BLL*.zip files, sort them, and process
find "$SOURCE_DIR" -name "BLL*.zip" -type f | sort | while read -r zipfile; do
    # Get the base name without path and extension
    basename_zip=$(basename "$zipfile" .zip)
    
    # Remove duplicate indicators (1), (2), etc. to get the original name
    clean_name=$(echo "$basename_zip" | sed 's/ *([0-9]\+) *$//')
    
    # Create a temporary directory for this extraction
    temp_dir=$(mktemp -d)
    
    echo "Processing: $zipfile"
    
    # Extract to temporary directory
    if unzip -q "$zipfile" -d "$temp_dir"; then
        # Move contents to destination, avoiding duplicates
        find "$temp_dir" -mindepth 1 -maxdepth 1 -type d | while read -r folder; do
            folder_name=$(basename "$folder")
            dest_folder="$DEST_DIR/$folder_name"
            
            if [ ! -d "$dest_folder" ]; then
                echo "  Copying folder: $folder_name"
                mv "$folder" "$dest_folder"
            else
                echo "  Skipping duplicate folder: $folder_name"
            fi
        done
        
        # Handle any loose files in the zip root
        find "$temp_dir" -mindepth 1 -maxdepth 1 -type f | while read -r file; do
            file_name=$(basename "$file")
            if [ ! -f "$DEST_DIR/$file_name" ]; then
                echo "  Copying file: $file_name"
                mv "$file" "$DEST_DIR/"
            else
                echo "  Skipping duplicate file: $file_name"
            fi
        done
    else
        echo "  Error extracting $zipfile"
    fi
    
    # Clean up temporary directory
    rm -rf "$temp_dir"
done

echo "Extraction complete!"