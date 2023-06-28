
# Fake News Detection Application

This project is an application developed for detecting fake news. It consists of two backend APIs: one serves as a mock task and the other functions as the main API.

## How It Works

1. First, you need to run the main backend API. It runs the trained models on a Flask server and sends a response to the mock API.

2. Then, you need to run the mock API. It works with Node.js and sends the user's input text to the Flask server.

3. When the user enters the desired news and clicks the "Submit" button, the text is sent to the mock API. The mock API forwards the text to the Flask server, which feeds it into the trained model and returns the output back to the mock API. Finally, the mock API sends the output to the user.

Note that, if you want to try your own model use the same code instruction in fakenewsdetection.py
## Installation

You can set up the project on your local machine by following these steps:

1. Clone this repository:
git clone [https://github.com/theprelior/fake-news-detector-api.git]


2. To run the main backend API, follow these steps:
- Go to the `main_api` directory:
  ```
  cd main_api
  ```
- Install the required packages 
  ```
- Start the Flask server by running the following command:
  ```
  python lastconnectedversion.py
  ```

3. To run the mock API, follow these steps:
- Go to the `mock_api` directory:
  ```
  cd mock_api
  ```
- Install the required packages by running the following command:
  ```
  npm install
  ```
- Start the mock API by running the following command:
  ```
  node index.js
  ```

4. Access the application in your browser at `http://localhost:3000`.


## License

This project is licensed under the [MIT License](LICENSE).
