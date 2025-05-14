# ABC-Stacking-for-Subglacial-Lakes
This project is a stacking ensemble learning system optimized by the Artificial Bee Colony (ABC) algorithm, specifically designed to automatically predict subglacial lakes (SLs) beneath the Antarctic ice sheet using radio-echo sounding (RES) data. The proposed method demonstrates excellent performance under extremely imbalanced data conditions, significantly improving the detection capability of subglacial lakes. It has been successfully applied to the Gamburtsev Subglacial Mountains in East Antarctica.

📚Project Structure

Dataset - how to download the data
Feature exteaction - how to define features
Stacking model - how to use abc-stacking model 

📦 Installation

pip install numpy pandas scikit-learn xgboost lightgbm catboost tqdm joblib

📂 Data Preparation

After download the dataset and use the make_feature.m
Ensure the following CSV files are placed in the training/ directory:
- dataset_ratio_1_to_***_0.csv – Training dataset. 
- all_w_validation.csv – Validation and test dataset.
  
CSV Column Requirements:
- Features: TFF, CBRP, Basal elevation, Hydraulic head gradient, Roughness
- Label: Label
Please create the training and validation datasets according to your specific needs or based on the data distribution described in my paper. 

🛠 Usage
python abc_stacking.py
This will:
1. Load the datasets.
2. Run ABC optimization to select the best stacking configuration.
3. Train the final stacking model.
4. Evaluate the model on the test set and output the F1 score.
5. Save the trained model as abc_stacking_model.pkl.
   
📧 Contact
For questions or suggestions, please contact: qianma@tongji.edu.cn without any hesitation.

📄 License and Usage Notice
This project is released under the MIT License; however, it is currently under peer review.
- 📖 This version is provided solely for academic review and evaluation purposes during the peer review process. The full version of the code will be fully released after acceptance.
I am also preparing and organizing the corresponding feature result plots for each listed lake. 
- ❗ Please do not  redistribute, or use the code for any commercial purposes at this stage.
- 📅 The code will be fully and officially open-sourced under the MIT License upon completion of the peer review.
  
📜 MIT License

Copyright (c) 2025 Qian M.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.

