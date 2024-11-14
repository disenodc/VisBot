![Logo visbot](https://raw.githubusercontent.com/disenodc/visbot/main/bot_1.png)
# VisBot - Visualization Recommender with iA

- Visitar: [Sitio Web](https://visbot.streamlit.app/) / [LICENCIA](https://github.com/disenodc/visbot/blob/main/LICENSE)
- Author: BCCs. Luis Dario Ceballos. (2024). *VisBot - Visualization Recommender with iA*. 
- PhD in Computer Science. UAI (Inter-American Open University), Buenos Aires-Argentina.
   CESIMAR - CENPAT - CONICET. Puerto Madryn, Chubut, Argentina, (U9120).


## PIPELINE locally executed

- Clone the repository

```bash
git clone https://github.com/disenodc/visbot.git
```
- Navigate to directory

```bash
cd ruta/al/directorio/del/repositorio/
```

- then install required libraries

```bash
 pip install -r requirements.txt
```
- Finally run the application

```bash
 streamlit run visbot.py
```

/////////////////////////////////
## PROJECT


1. Summary

In the era of big data, the ability to transform large volumes of data into understandable and actionable information is crucial. Data visualizations play an essential role in this process, allowing you to identify patterns, trends, and anomalies intuitively. However, creating effective visualizations can be challenging, especially for users without experience in data analysis or graphic design. With the emergence of Large Scale Language Models (LLMs), such as those developed by OpenAI, a new possibility opens up for the automation of this process. This project proposes the construction of an interactive interface using the Streamlit framework, which is supported by LLMs and machine learning techniques to analyze the structure of the data provided by users and generate recommended visualizations. This tool has the potential to democratize access to advanced data analytics, making data interpretation easier for a wide range of users.

 2. Objectives

General Objective:

Develop an interactive interface based on Streamlit, using LLMs and machine learning techniques, to automatically analyze and generate recommended data visualizations, adapted to the data provided by users.

Specific Objectives:

   1. Integrate an LLM model that can interpret data descriptions and contextualize their structure.
   2. Develop machine learning algorithms that allow selecting the most appropriate type of visualization for the data in question.
   3. Implement an interface in Streamlit that facilitates user interaction with the system, allowing the loading of data and the visualization of the generated recommendations.
   4. Evaluate the effectiveness of automatically generated visualizations compared to those created manually by experts.
   5. Validate the tool through tests with users of different levels of experience in data analysis.


3. Development

The development of the project will take place in several key stages:

3.1. Selection and Configuration of the LLM:
   - A suitable OpenAI LLM will be selected, with the ability to understand and process natural language instructions related to data structure and analysis.
   - The model will be configured so that it interacts efficiently with the data entered by users and can suggest visualizations based on recognizable patterns in the data.

3.2. Design of Recommendation Algorithms:
   - Machine learning algorithms will be developed that analyze the structure of the data (types of variables, relationships between them, etc.) to recommend the most appropriate visualization.
   - These algorithms will be trained using standard data sets to ensure the accuracy of the recommendations.

3.3. Construction of the Interface in Streamlit:
   - An interactive interface will be designed in Streamlit, where users will be able to upload their data, receive recommendations and generate visualizations in an automated manner.
   - The interface will be intuitive and accessible, designed for users with various levels of technical experience.

3.4. Integration and Testing:
   - All system components will be integrated, ensuring that communication between the LLM, the machine learning algorithms and the interface is fluid.
   - Tests will be carried out with users to identify possible improvements in the usability and accuracy of the system.

 4. Methodology

The project methodology will be based on an agile, iterative approach, which allows rapid adjustments in response to user feedback and results obtained during testing. Stages of the methodology:

- Initial Research: Review of the literature on LLMs, machine learning and data visualization techniques.
- Technological Development: Programming and integration of components using Python, Streamlit, and OpenAI APIs.
- Algorithm Training: Use of test datasets to train and adjust recommendation models.
- Validation and Evaluation: Application of usability and precision metrics to evaluate the developed tool.
- Documentation and Presentation: Preparation of technical and academic documentation to present the findings and operation of the system at the congress.

 5. Expected Results

At the end of the project, it is expected to have developed a tool that:

- Provide accurate and useful visualizations based on automated data analysis.
- Facilitate the understanding and presentation of data for users with different levels of technical experience.
- Democratize access to advanced analytics tools, allowing more people to make informed decisions based on their data.
- Create a positive impact on the academic and professional community by presenting an innovative approach to data analysis automation.


 6. Conclusion

This project proposes an innovative solution for one of the contemporary challenges in data management and visualization. By combining the power of LLMs with advanced machine learning techniques and an accessible interface, it can offer a powerful tool to improve the understanding of complex data. Automating visualization generation not only saves time, but also increases the accuracy and relevance of visual presentations, benefiting a wide range of users. The implementation of this system has the potential to mark a significant advance in the accessibility and effectiveness of data analysis.

 7. References

 - Brown, T., Mann, B., Ryder, N., et al. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems, 33*, 1877-1901.
- McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 51-56.
- OpenAI. (2024). *OpenAI API documentation*. Recuperado de https://platform.openai.com/docs
- Plotly. (2024). *Plotly Python Graphing Library*. Recuperado de https://plotly.com/python/
- Reitz, K., & Team. (2024). *Requests: HTTP for Humans*. Recuperado de https://requests.readthedocs.io/
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135-1144.
- Seaborn: Statistical Data Visualization — Seaborn 0.11.1 Documentation. (2020). Recuperado de https://seaborn.pydata.org/
- Streamlit. (2024). *The fastest way to build and share data apps*. Recuperado de https://streamlit.io/
- The Pandas Development Team. (2024). *Pandas documentation*. Recuperado de https://pandas.pydata.org/

