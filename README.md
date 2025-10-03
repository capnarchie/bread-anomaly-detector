##  Quick Start

### 1. Setup Environment
```bash

# use existing virtual environment
source .venv/bin/activate  # Linux/Mac
# .venv\\Scripts\\activate  # Windows
```

### 2. Run Detection System
```bash
# Use the main entry point
python run_detection.py

# Or directly run the detection script
python src/improved_padim_deploy.py
```

### 3. Train New Model (if needed)
```bash
# First organize your data
python utils/data_organizer.py

# Then train the model
python src/improved_train_padim.py
```
