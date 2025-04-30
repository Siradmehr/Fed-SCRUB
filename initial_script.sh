create_dir() {
  if [ ! -d "$1" ]; then
    mkdir -p "$1"
    echo "Created directory: $1"
  else
    echo "Directory already exists: $1"
  fi
}

echo "Setting up project directories..."

# Create the necessary directories
create_dir "checkpoints"
create_dir "data"

echo -e "\nInstalling requirements..."
if [ -f "requirements.txt" ]; then
  pip3 install -r requirements.txt
  echo "Requirements installed successfully."
else
  echo "Warning: requirements.txt file not found."
fi

echo -e "\nSetup complete! Created basic directories and installed requirements."