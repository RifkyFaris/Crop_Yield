# 🌾 Crop Yield Prediction System

Agriculture remains one of the most crucial sectors for Sri Lanka’s economy and food security. However, farmers often face challenges in forecasting crop yields due to fluctuating weather conditions, inconsistent rainfall, soil fertility variations, and limited access to analytical tools. Traditional forecasting methods—based largely on experience and assumptions—are frequently unreliable, resulting in financial losses, inefficient resource use, and environmental degradation.

To address these issues, this project introduces a **data-driven crop yield prediction system** powered by **Data Mining and Machine Learning (ML)** techniques. The system analyzes agricultural and environmental data, such as rainfall, soil type, temperature, fertilizer usage, and irrigation practices, to generate accurate yield forecasts.

The project’s primary goal is to assist **farmers, agricultural researchers, consultants, and policymakers** in making informed decisions that improve productivity, optimize resource management, and enhance food security.

Using regression-based ML algorithms (Linear Regression, Lasso, and Decision Tree), the system learns from real agricultural datasets to predict yield outcomes for different crops under varying environmental conditions. The predictions are made accessible through a **Flask-based web interface**, where users can input parameters and receive real-time yield estimates.

Beyond prediction, the system includes **data visualization features**, enabling users to observe trends such as how rainfall or fertilizer levels affect yields. This visual insight helps users interpret model outcomes intuitively and supports data-driven agricultural planning.

Ultimately, the Crop Yield Prediction System demonstrates how technology and analytics can transform traditional farming practices into **smart agriculture**, fostering sustainable growth and resilience in the agricultural sector.

# 🛠️ Tech Stack – Crop Yield Prediction System

The **Crop Yield Prediction System** is developed using a combination of data science, machine learning, and web technologies to ensure scalability, accuracy, and usability.

---

## 💻 Programming & Development

* **Language:** Python 3.11
* **Development Environment:** Jupyter Notebook, Visual Studio Code

---

## 🤖 Machine Learning & Data Science Libraries

* **Scikit-learn:** For implementing regression algorithms (Linear Regression, Lasso, Decision Tree) and evaluating model performance.
* **Pandas:** For data manipulation, cleaning, and preprocessing.
* **NumPy:** For numerical computation and matrix operations.
* **Matplotlib:** For data visualization and chart creation (e.g., yield vs rainfall).

---

## 🌐 Web Framework

* **Flask:**

  * Backend web framework used to build the predictive web interface.
  * Handles user inputs (e.g., rainfall, soil type, fertilizer usage) and serves prediction results dynamically.

---

## 💾 Model Storage & Serialization

* **pickle:** Used to serialize and save trained ML models (`model.pkl`).
* **json:** For storing configuration and structured data exchange between backend and frontend.

---

## 📈 Data & Visualization Tools

* **Matplotlib / Seaborn:** To visualize trends, correlations, and prediction outputs.
* **Exploratory Data Analysis (EDA):** Conducted through Jupyter Notebook to uncover key patterns in the dataset.

---

## ☁️ Dataset

* **Source:** [Hugging Face – Crop Yield Dataset](https://huggingface.co/datasets/sydniezhao/crop_yield)
* **Records:** 10,000+ agricultural entries
* **Features:** Rainfall, soil type, temperature, fertilizer usage, crop type, irrigation level, weather patterns

---

## 🧰 Additional Tools

* **Git & GitHub:** For version control and collaborative development.
* **Microsoft Excel:** For manual inspection and validation of dataset attributes.
* **Browser:** For testing Flask UI (Chrome/Edge).

---

This tech stack ensures the project’s reliability, modular design, and ease of deployment, allowing smooth integration between **machine learning models** and **user-facing web applications**.
