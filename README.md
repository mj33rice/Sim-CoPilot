# SimCoPilot: Evaluating Models for Co-pilot-Style Code Generation
**SimCoPilot** is a benchmark with:
- **Purpose**: Evaluate LLMs as interactive coding assistants in "copilot"-style.
- **Focus**: Test AI's ability to integrate and complete code in complex software environments.

![Figure 1: Workflow for each of the 1,163 programming tasks in SIMCOPILOT.](figures/Workflow.png)
🤔*Question: Can an AI tool correctly complete code snippets like `method body`, `if-statements`, or `for-loops` from real-world projects?*

**SimCoPilot Demo**
<p align="center">
  <img src="./figures/SimCoPilot_hd.gif" alt="SimCoPilot Demo"/><br>
  <!-- <em>Figure 2: SimCoPilot Demo</em> -->
</p>
<!-- **SimCoPilot Demo**
![Figure 2: SimCoPilot Demo](./figures/SimCoPilot_hd.gif) -->


**Why SimCoPilot?**
- **Real-World Complexity**: Tests AI on complex, real-project tasks, not just concise and standalone programs.
- **Real-Code Understanding**: Focuses on AI's ability to work with actual code, without needing manually annotated problem descriptions to prompt for each task.
- **Fine-grained Results**: Stratifies results according to metrics such as distance to the nearest referenced object, proximity to the nearest comment, and various programming constructs.

## 📊 Dataset


<!-- The data for this project can be found in the ` dataset/SimCoPilot.csv.zip` file.  -->

**Hosting, Licensing, and Maintenance Plan.**
- **Dataset and Metadata Access.** The data for this project can be found in the `dataset/SimCoPilot.csv.zip` file. We commit to maintaining the dataset with regular updates and revisions to correct any issues and integrate new contributions. The dataset and its associated metadata, nutrition labels, documented using the Croissant metadata framework will be released.
<!-- - **Dataset and Metadata Access.** The dataset and its associated metadata, documented using the Croissant metadata framework, can be viewed and downloaded at [Huggingface Datasets:SimCoPilot](https://huggingface.co/datasets/mj33/SimCoPilot). -->
<!-- The data nutrition label can be found at [Data Nutrition Label](https://github.com/mj33rice/SimCoPilot/tree/main/dataset#data-nutrition-label). -->
- **Licensing:** The data is shared under the [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) and code is licensed under MIT License.
- **Maintenance Plan:** We commit to maintaining the dataset with regular updates and revisions to correct any issues and integrate new contributions. Updates will be documented in the repository's release notes section.

<details>
<summary><b>Additional Support Data</b></summary>

The extended dataset and supplementary materials can be downloaded from our [Google Drive repository](https://drive.google.com/drive/folders/1FCyPWQO3cQKxtJbQIGWzaE6G4tE5cmJa?usp=sharing).

**Download Instructions:**
1. Access the Google Drive link above
2. Download the required files
3. Extract and place the downloaded files in `./example_code/Python/Image_Filtering/` directory
4. Ensure the directory structure matches the expected paths in the code examples

```bash
# Create directory if it doesn't exist
mkdir -p ./example_code/Python/Image_Filtering/

# Place downloaded files in the directory
cp /path/to/downloaded/files/* ./example_code/Python/Image_Filtering/
```
</details>

## 🏆 LeaderBoard
| Model                          | Python Infill     | Python Completion    | Java Infill       | Java Completion       | HumEval |
|--------------------------------|-------------------|----------------------|-------------------|-----------------------|--------|
| GPT-4o (2024-08-06)            | 78.5±4.1          | **69.9±6.2**         | 78.1±4.9          | 54.9±5.7              | 92.7   |
| o3-mini (high)                 | **83.3±3.7**      | 66.0±6.3             | **87.6±3.8**      | 59.1±5.7              | —      |
| Claude 3 Opus                  | 75.1±4.3          | 67.4±6.2             | 66.8±5.5          | 69.2±5.4              | 84.9   |
| Claude 3.7 Sonnet (ET.)        | 80.9±3.9          | 67.5±6.4             | 86.9±4.0          | 68.5±5.4              | **97.8** |
| Claude 3.7 Sonnet              | 74.3±4.4          | 57.1±6.6             | 71.0±5.3          | **69.5±5.3**          | 94.9   |
| Claude 3.5 Haiku               | 70.9±4.5          | 60.4±6.5             | 66.4±5.5          | 61.2±5.6              | 88.1   |
| Llama 3.3 70B                  | 58.1±4.9          | 53.7±6.8             | 65.7±5.5          | 49.3±5.8              | 88.4   |
| Llama 3.1 8B                   | 43.7±4.9          | 39.6±6.6             | 36.4±5.7          | 32.8±5.4              | 72.6   |
| DeepSeek-R1 671B               | 73.3±4.4          | 64.6±6.4             | 77.4±4.9          | 59.4±5.6              | 97.7   |
| R1-Distill-Qwen14B             | 46.9±5.0          | 38.7±6.5             | 38.9±5.7          | 39.1±5.6              | —      |
| Qwen2.5-Coder-32B              | 70.2±4.5          | 64.7±6.4             | 63.2±5.6          | 56.3±5.7              | 92.1   |
| Qwen-QwQ-32B                   | 52.6±5.0          | 47.2±6.7             | 51.6±5.8          | 34.3±5.4              | 97.6   |
> **Note**: To ensure consistency, the AI-for-code model's randomness is minimized, aiming for the most likely/preferred outcomes. All SIMCOPILOT benchmark results are reported with 95% confidence intervals.
## 🚀 Getting Started

1. Clone the repository.
2. Install necessary dependencies.
3. Run the analysis on your target codebase by specifying the file path and dependency range parameters.

To install the Dependency Analyzer, clone this repository and run the setup script:

```bash
git clone git@github.com:mj33rice/Sim-CoPilot.git
pip install -r requirements.txt
```

<details>
<summary><b>Model Inference API Setup</b></summary>
1. Install the necessary Python packages:

```bash
pip install anthropic together
```
2. Open your terminal and type the following command:
```bash
nano ~/.bash_profile 
```
If you’re using a newer version of macOS, you might need to use `~/.zshrc` instead:
(or nano ~/.zshrc if you’re using a newer version of macOS)
```bash
nano ~/.zshrc
```
3. Add the following line to the file, replacing `your-api-key-here` with your actual API key:
```bash
 export ANTHROPIC_API_KEY='your-api-key-here' 
```
 If you're using OpenAI, use this line instead:
```bash
 export OPENAI_API_KEY='your-api-key-here'
```
If you're using platform such as Together AI for open source model inference, use this line instead:
```bash
export TOGETHER_API_KEY='your-api-key-here'
```

4. Save the file and exit the editor (press `Ctrl+O`, then `Enter`, then `Ctrl+X`)
5. Load the updated profile by running: 

```bash
source ~/.bash_profile (or source ~/.zshrc)
```
</details> 


## 🏃How to Run

### Code Generation 

The commands below enable code generation model execution on Java and Python files for both closed-source and open-source models. Specify source code paths, test cases, the model, and the code generation mode as follows:

### Model Inference
```python
python model_inference.py <source_code_path> <test_cases_path> --gen_model <model_name> --code_gen_mode <mode>
```

- `source_code_path`: Path to the Python or Java source code file. 
- `test_cases_path`: Path to the JSON test cases file.
- `--gen_model`: Close source model for code generation, e.g.`gpt-4o-2024-08-06`, `claude-3-7-sonnet-latest`.
- `--code_gen_mode`: Specifies the code generation task type:
    - `with_afterlines`: For infilling tasks with context before and after the target code.
    - `no_afterlines`: For completion tasks, generating code to finish a block without subsequent context.
  

<details>
<summary><b>Example Command</b></summary>

Code generation script with specific parameters, you can use the following command:
This command specifies the use of the `gpt-4o-2024-08-06`models for code generation with the mode set to `with_afterlines` indicating that the generation should consider both the preceding and following context.
```python
#Specifiy Model to Run
python model_inference.py \
./example_code/Python/simplex_method/simplex_method.py \
./example_code/Python/simplex_method/simplex_method.json \
--read_dependency_results --update_def_line \
--gen_model gpt-4o-2024-08-06 \
--code_gen_mode with_afterlines
```
</details> 


<details>
<summary><b>Run from the Script</b></summary>

```bash
#For Python tasks
chmod +x run_python_paral.sh
./run_python_paral.sh

#For Java tasks
chmod +x run_java_paral.sh
./run_java_paral.sh
```
</details>


## 🛠 Post Processing 
```python
python -m helper_functions.update_post_process_and_eval ./PATH/to/result_folder
```

#### Example of Post-Processing

For detailed examples of code Post-Processing, please refer to the figure below:

<p align="center">
  <img src="figures/Python_infill_post_processing_example.png" alt="Example infill task from SimCoPilotP and step-by-step post-processing demonstration."/><br>
  <em>Figure 2: Example infill task from SimCoPilotP and step-by-step post-processing demonstration.</em>
</p>

<p align="center">
  <img src="figures/Java_infill_post_processing_example.png" alt="Example completion task from SimCoPilotJ and step-by-step post-processing demonstration"/><br>
  <em>Figure 3: Example completion task from SimCoPilotJ and step-by-step post-processing demonstration</em>
</p>

<!-- ![Figure 2: Example infill task from SIMCOPILOTP and step-by-step post-processing demonstration.](figures/Python_infill_post_processing_example.png)

*Figure 2: Example infill task from SimCoPilotP and step-by-step post-processing demonstration.*

![Figure 3: Example infill task from SIMCOPILOTP and step-by-step post-processing demonstration.](figures/Java_infill_post_processing_example.png)

*Figure 3: Example completion task from SimCoPilotJ and step-by-step post-processing demonstration* -->

## 🔍 Stratified Evaluation
Detailed results comparing the test case pass ratios of various LLMs:  
- Categorized by models and different programming constructs: [code](helper_functions/code_gen_result_display.ipynb)

<!-- ![Figure 4: Python Infill & Completion - Pass Rates by Program Constructs](./figures/Python_Construct_v2_jpeg.jpg)
*Figure 4: Python Infill & Completion - Pass Rates by Program Constructs*

![Figure 5: Java Infill & Completion - Pass Rates by Program Constructs](./figures/Java_Construct_v2_jpeg.jpg)
*Figure 5: Java Infill & Completion - Pass Rates by Program Constructs* -->

<table>
  <tr>
    <td align="center" colspan="2">
      <img src="./figures/Python_Construct_v2_jpeg.jpg" alt="Python Infill & Completion" style="width: 100%; height: auto;"/><br>
      <em>Figure 4: Python Infill & Completion - Pass Rates by Program Constructs</em>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="./figures/Java_Construct_v2_png_compressed.png" alt="Java Infill & Completion" style="width: 100%; height: auto;"/><br>
      <em>Figure 5: Java Infill & Completion - Pass Rates by Program Constructs</em>
    </td>
  </tr>
</table>

<p align="center">
  <!-- <em>Pass Rates by Program Constructs</em> -->
</p>

- Categorized by distance to the nearest referenced object: [code](helper_functions/horizon_dist.ipynb)

<p align="center">
    <img src="./figures/Pass_rate_group_by_ref_dist_w_errbar_compressed.png" alt="Pass Rates by Distance to Referenced Object"/><br>
    <em>Figure 6: Pass Rates by Distance to Referenced Object</em>
</p>

<!-- ![Figure 6: Pass Rates by Distance to Referenced Object](./figures/Pass_rate_group_by_ref_dist_w_errbar_compressed_jpeg.jpg)
*Figure 6: Pass Rates by Distance to Referenced Object* -->

- Categorized by proximity to the nearest comments: [code](helper_functions/get_comment_dist.ipynb)

<p align="center">
  <img src="./figures/group_by_comment_dist_w_errbar_compressed.png" alt="Pass Rates by Distance to Closest Comment"/><br>
  <em>Figure 7: Pass Rates by Distance to Closest Comment</em>
</p>
<!-- ![Figure 7: Pass Rates by Distance to Closest Comment](./figures/group_by_comment_dist_w_errbar_compressed_jpeg.jpg)
*Figure 7: Pass Rates by Distance to Closest Comment* -->

- Error Analysis
<p align="center">
  <img src="./figures/Error_Analysis_compressed_600_jpeg.jpg" alt="Error Analysis"/><br>
  <em>Figure 8: Cumulative Output by Category</em>
</p>
<!-- ![Figure 8: Cumulative Output by Category](./figures/Error_Analysis_compressed_600_jpeg.jpg)
*Figure 8: Cumulative Output by Category* -->

- **Model Size vs. Error Rates:** Larger or more sophisticated models generally have fewer errors but don't consistently show lower error rates across all categories.
- **Common Errors:** Compilation and syntax errors are prevalent across most LLMs, indicating challenges in understanding code structure or syntax in code generation tasks.

These observations highlight that while model size often correlates with performance, specific error types reveal unique strengths and weaknesses in each model's understanding of code structure and syntax.

## 📧 Contact Us 

For any inquiries or further information, please feel free to reach out to us at [mj33@rice.edu](mailto:mj33@rice.edu).

## License

The data is shared under the [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) and code is licensed under MIT License.
