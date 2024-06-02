import tensorflow as tf

str1=input("Enter the sentence:")

model=tf.keras.models.load_model('D:\Python\ML\Projects\Sarcasm\model_sarcasm')

model_probs=model.predict([str1])

model_preds=tf.round(model_probs)

if model_preds==1:
    print(f"the given sentence {str1} is found to be sarcastic")
else:
    print(f"The given sentence {str1} is found to be not a sarcastic")

