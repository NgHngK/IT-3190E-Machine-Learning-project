---
title: ML Demo2025
emoji: 🐠
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Overview
In this project, we develop multiple machine learning models to predict gas prices using historical and market data.

The project is deployed using Gradio and containerized with Docker for easy reproducibility and deployment.

For details, please look at the report named as "ML_report.pdf".

# Getting started 
To run the project locally, follow the steps below:

1. Restart docker
<pre>
  sudo systemctl restart docker
</pre>

2. Build the docker image
<pre>
  sudo docker build -t oil-price-predictor .
</pre>

3. Run the docker container
<pre>
  sudo docker run -p 7860:7860 oil-price-predictor
</pre>

Once running, the localhost link will appear in your terminal. Open it in your browser to access the application.

Note: These instructions are tailored for Ubuntu. Adjust accordingly if using another operating system. Also, the docker cannot be built when using workspace wifi/ school connection.

# Acknowledgement
This project is built by a group of students at Hanoi University of Science and Technology, Vietnam. Thanks for the contribution from Hoang Trung Dung, Le Minh Kiet, Le Tam Quang, Nguyen Vu Thuy, Nguyen Tat Hung. These authors contributed equally. The order is arbitrary.
