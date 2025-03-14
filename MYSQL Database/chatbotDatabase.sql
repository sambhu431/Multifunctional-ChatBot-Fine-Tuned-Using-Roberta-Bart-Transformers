

SHOW DATABASES;

USE chatbotDatabase;

SHOW TABLES;

SELECT * FROM UserInputs;


SELECT input_text, bot_response FROM UserInputs;


ALTER TABLE UserInputs
MODIFY bot_response VARCHAR(1000);


