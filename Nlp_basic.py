import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
import pandas as pd
import numpy as np
import datetime
import re
# from allennlp.predictors.predictor import Predictor
# import allennlp_models.coref



preditor = Predictor.from_path("http://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
preditor


predict = preditor.predict(document="A 2year old boy is hit by the black SUV on Tuesday 5:30pm near Kindergarden The Car is driven by his neighbor")
docs = "A 2year old boy is hit by the black SUV on Tuesday 5:30pm near Kindergarden The Car is driven by his neighbor"
Tspan = []
for cvx in predict['clusters']:
  for part in cvx:
    Tspan.append(part)


displacy.render(nlp(docs),style='ent',jupyter=True)


kc10= "A two-year-old child is hit by the Black SUV on Tuesday 5:30pm near kindergarden The truck is driven by his neighbor"
kc11 ="A one-year-old toddler is hit by the white BMW on Tuesday 4:30pm near varanda The car is driven by his father"
data = nlp(kc10)
print(data)
for d in data:
  print(d,d.dep_,d.pos_)
for sd in data.ents:
  print(sd.text,sd.label_)



Model = tf.keras.Sequential([tf.keras.layers.Dense(units=6,input_shape=[1]),
tf.keras.layers.Dense(units=6),tf.keras.layers.Dense(units=1)])
Model.compile(loss="mean_squared_error",optimizer=tf.keras.optimizers.Adam(0.1))

Model.fit(output_data,train_data,epochs=5)


output_data = [8,24,56,64,112,220,22500]
train_data = [4,12,28,32,56,110,150]


Model.predict([8])

data1 = nlp(kc11)
for dt in data1:
  print(dt,dt.dep_,dt.pos_)
for fc in data1.ents:
  print(fc.text,fc.label_)



victim_tokens =  ["child",'baby',"toddler"]
vehicle_tokens = ["car","Van","truck"]
matcher = Matcher(nlp.vocab)
victim_pattern = [[{"lower":r}] for r in victim_tokens]
vehicle_pattern = [[{"lower":r}] for r in vehicle_tokens]
matcher.add("VICTIM",None,*victim_pattern)
matcher.add("VEHICLE",None,*vehicle_pattern)


mat = matcher(data1)
span = lambda x:next(filter(lambda y:y[0] == nlp.vocab.strings[x],mat),False)
tan = span("VEHICLE")
veh = data1[tan[1]:tan[2]]
vehicles = next(filter(lambda x:x.dep_ == "nsubjpass" and x.pos_ == "NOUN",veh.subtree),False)
vehicles = next(filter(lambda x:x.dep_ == "ROOT" and x.pos_ == "VERB",vehicles.ancestors),False)
by = next(filter(lambda x:x.dep_ == "agent" and x.pos_ == "ADP",vehicles.children),False)
Accused = next(filter(lambda x:x.dep_ =="pobj" and x.pos_ == "NOUN",by.children),False)
Accused


def Frames(model):
  matches = matcher(model)
  get_span = lambda x:next(filter(lambda y:y[0] == nlp.vocab.strings[x],matches),False)
  Victim = get_span("VICTIM")
  Vehicle =get_span("VEHICLE")
  victim = model[Victim[1]:Victim[2]]
  vehicle = model[Vehicle[1]:Vehicle[2]]
  victimm = next(filter(lambda x: x.dep_ == 'nsubjpass' and x.pos_ == 'NOUN',victim.subtree))
  maintree =list(list(victimm.ancestors)[0].subtree)
  Model = next(filter(lambda x:x.dep_ =="pobj" and x.pos_ =="PROPN",maintree))
  Time = next(filter(lambda x:x.dep_ =="pobj" and x.pos_ =="NUM",maintree))
  Place = list(filter(lambda x:x.dep_ =="pobj" and x.pos_ == "PROPN",maintree))[1]
  Color = list(filter(lambda x:x.label_ =="ORG",model.ents))
  color = Color[0].text.split(" ")[1]
  vehicles = next(filter(lambda x:x.dep_ =="nsubjpass" and x.pos_ == "NOUN",vehicle.subtree),False)
  vehicles = next(filter(lambda x:x.dep_ == "ROOT" and x.pos_ =="VERB",vehicles.ancestors),False)
  by = next(filter(lambda x:x.dep_ == "agent" and x.pos_ == "ADP",vehicles.children),False)
  Accused = next(filter(lambda x:x.dep_ =="pobj" and x.pos_ =="NOUN",by.children),False)
  dicts = {"Victim":victim[0],
           "Model":Model,
           "Time":Time,
           "Place":Place,
           "color":color,
           "vehicle":str(vehicle),
           "Accused":str(Accused)
           }
  return dicts



docker = [data,data1]
final =[]
for neural in docker:
  dt = datetime.datetime.now()
  print(dt.strftime("%d-%m-%Y %H-%M-%S"))
  final.append(Frames(neural))



finaldf= pd.DataFrame(final)
finaldf
