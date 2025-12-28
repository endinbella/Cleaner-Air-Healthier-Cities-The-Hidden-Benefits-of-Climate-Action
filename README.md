# Cleaner Air, Healthier Cities: The Hidden Benefits of Climate Action

This project uses data from the **UK Co-Benefits Atlas** to show how climate action can improve **air quality and public health** across the United Kingdom. It combines **Level 1 spatial data**, which represent different administrative regions, and **Level 2 indicators** that measure air quality co-benefits over time. The analysis covers the period **2025–2050** and highlights that while climate action brings real health benefits, these benefits are **not experienced equally** by all regions.

## Project Workflow & Tools

The analysis and visualisations were developed using **Python** with a clear and structured workflow. The main datasets are stored in the `dataset` directory as `Level_1.xlsx` and `Level_2.xlsx`. Initial exploration and calculations were carried out in **Jupyter Notebook (`visdat.ipynb`)**, followed by further processing and visual generation using Python scripts. The results are presented through a simple **web-based dashboard** built with `dashboard.py` and `index.html`, supported by visual assets stored in the `assets` folder.

## Analytical Approach

Using Level 1 data, the project compares air quality benefits between regions, while Level 2 data are used to calculate key indicators such as **total long-term benefits (sum)**, **trends over time (slope)**, and **benefit stability (gap)**. These indicators help explain not only how much improvement occurs, but also how consistent and sustained the benefits are across different regions.

## Key Findings

The visualisations show that most regions experience **moderate improvements in air quality**, with relatively stable patterns over time. When benefits are aggregated over the long term, clearer **regional differences** emerge. By combining trend and stability indicators in a **quadrant analysis**, the project highlights areas that may require more **focused and targeted climate action**. Overall, the findings emphasize the importance of **place-based climate policies** to ensure that the health benefits of cleaner air are shared **more fairly and sustainably**.
