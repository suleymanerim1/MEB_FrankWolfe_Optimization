# MEB_FrankWolfe_Optimization

I have put neccessary information about project guidelines (taken from both mail and moodle) into "Project Descripton.txt"

Lecture notes about Frank Wolfe in folder
Papers given by prof in folder

I only added Frank Wolfe for now, but if you think there are some other sources necessary to understand papers, you can add them. Please add only necessary pages to continue focused on topic.


# HOW TO CREATE ENVIRONMENT

Update conda
"conda update conda -y"

From terminal type the line given below. It will create base python environment.
"conda create -p venv python==3.10 -y"

Now activate environment from terminal
"conda activate venv/"

Install all necessary packages and setup project. The code will create our custom packages (src)
Also it will download necessary packages in requirements.txt file
(important point : if you already did this before in our project, then comment out last line of reqiurements.txt (-e .) .
So that you only install new packages but do not create our src packages unnecesarly.

Now go to requirements.txt file and delete the comment symbol (#) before "-e ."
Then in terminal type:
"pip install -r requirements.txt"

If you need to install extra packages from pip, write it in requrirement.txt and use only code above "pip install -r requirements.txt"
So that we can also have the same package in our environment.

