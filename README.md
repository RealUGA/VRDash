This is a dashboard application for analyzing eye tracking data.
Sample eye tracking data is provided in the data folder.

It is recommended to run this project in a virtual environment.
To install, follow the commands below.

(Mac/Linux)
```
$ python3 -m venv venv
$ . venv/bin/activate
```

(Windows)
```
$ py -3 -m venv venv
\> venv\Scripts\activate
```

To exit virtual environment, enter deactivate in the CLI

To install all project requirements, enter the following commands:

```
$ pip install dash
$ pip install dash-bootstrap-components
$ pip install pandas
$ pip install dash-extensions
$ pip install opencv-python
```

To run the application, enter:

```
$ python app.py
```

The application can then be accessed by navigating to http://localhost:8050/
