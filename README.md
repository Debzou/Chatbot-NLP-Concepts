# Chatbot-NLP-Concepts

## installation

```bash
install python3.8
virtualenv venv --python=python3.8
pip install -r .\requirements.txt
pip freeze > requirements.txt
```
## process input

![Screenshot](/NLP_process.PNG)

## create the training model NN
(venv)

```
python training.py
```

the model will be save in data.pth

![Screenshot](/NN_diagram.PNG)

## run the chatbot

(venv)

Neural network chatbot

```
python chatbotNN.py
```

Tree chatbot

```
python chatbotTree.py
```

Now chat with your new friend !
