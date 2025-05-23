import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import random
#descarcam tokenizerul punkt pt a putea folosi word_tokenize
nltk.download('punkt')

# Tokenizare (impartirea unu text mai lung in curinte)
# și stemmare (trunchiarea unui cuvant pana la baza sa)
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    #facem un vector de lungime egala cu all_words, si il umplem cu zerouri
    #va reprezenta propozitia sub forma de bag of words
    bow = np.zeros(len(all_words), dtype=np.float32)
    #tokenizam propozitia si facem stemming pe fiecare cuvant
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    #se uita pe fiecare cuvant din toate cuvintele care apar in intents.json
    #iar daca unul din ele apare in propozitia curenta, pune 1 in bow
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bow[idx] = 1.0
    return bow

with open('intents.json', 'r') as f:
    intents_data = json.load(f)


#PREGATIM DATELE PENTRU ANTRENAREA MODELULUI
all_words = [] #vocabularul
tags = [] #etichete/intents
xy = [] #tuple token-tag
ignore_words = ['?', '!', '.', ',']

#mergem pe toate intent-urile din fisier
#preluam fiecare tag si il punem in vectorul de etichete
#token-izam fiecare pattern al tag-ului curent si punem toate cuvintele in vocabular
#salvam relatia intre tag si cuvintele din pattern-uri
for intent in intents_data['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokens = tokenize(pattern)
        all_words.extend(tokens)
        xy.append((tokens, tag))
 

#facem stemming pe vocabular si eliminam duplicatele (setul are doar val distincte)
all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))
tags = sorted(set(tags))

print(tags)
print(all_words)

#ne creem vectorii de antrenament: X-inputul, Y-outputul
X_train = [] #in x vom avea vectorul bag of words pentru feicare propozitie
             #care va contine doar valori de 1 si 0 reprezentant cuvintele din 
             #vocabular prezente in acea propozitie
y_train = [] #in y von avea tag-ul corespunzator fiecarei propozitii

#pentru fiecare tupla tokens-tag creem vectorul bag of words
#si il punem in x, iar tag-ul il punem in y (sub forma numerica, nu text)
for (pattern_sentence, tag) in xy:
    bow = bag_of_words(pattern_sentence, all_words)
    X_train.append(bow)
    label = tags.index(tag)
    y_train.append(label)

#transformam ambele array-uri in array-uri numPy pentru a fi compatibile cu pyTorch
X_train = np.array(X_train)
y_train = np.array(y_train)

#scopul acestei clase este ca transforme vectorii nostri de antrenament (X,y) 
#in date compatibile cu DataLoader-ul din pyTorch
#aceasta clasa extinde Dataset deci trebuie sa implementam metodele getitem si len
class ChatDataset(Dataset):
    def __init__(self):
        #len(X_train) = cate propozitii avem in total
        self.n_samples = len(X_train)
        self.x_data = X_train
        #convertim y_train la int64 fiindca asa cere pyTorch
        self.y_data = y_train.astype(np.int64)

    #returneaza un set de date de la un anumit index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index].squeeze()
    
    #returneaza numarul de propozitii
    def __len__(self):
        return self.n_samples
    
#clasa de baza pentru orice mdel pyTorch, extinde nn.Module
#va deveni reteaua noastra neuronala care clasifica intentiile chatbot-ului nostru
class NeuralNet(nn.Module):
    #se apeleasa la crearea obiectului si primeste numarul de neuroni 
    #(dimensiunea vectorului bag of words), cati neuroni vrem sa avem in 
    #straturile ascunse, si cate etichete vrem sa prezicem aka. cate intentii 
    #are chatbot-ul nostru
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        #creem straturile retelei neuronale
        #primul strat primeste neuronii initiali 
        #urmatoarele doua straturi intermediare primesc si returneaza hidden_size
        #neuroni, iar ultimul returneaza predictia
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, hidden_size)
        self.layer_4 = nn.Linear(hidden_size, num_classes)
        #functia de activare care daca primeste o valoare pozitiva o lasa asa
        #iar daca primeste o valoare negativa o face 0
        self.relu = nn.ReLU()

    #cum circula datele prin retea -> forwards
    # x = vectorul bag of words
    def forward(self, x):
        out = self.relu(self.layer_1(x))
        out = self.relu(self.layer_2(out))
        out = self.relu(self.layer_3(out))
        #la ultimul layer nu mai trecem prin relu deoarece vrem sa avem rezultatele nemodificate
        out = self.layer_4(out)
        return out
    

#ANTRENAREEEEEEE
batch_size = 8 #modelul invata din 8 modele de-odata
hidden_size = 16 #neuronii ascunsi pentru straturile intermediare
output_size = len(tags) #numarul de clase de iesire posibile este egal cu nr. de tag-uri
input_size = len(X_train[0]) #dimensiunea inputului = dimensiune bag of words
learning_rate = .001 
num_epochs = 1000 #parcurgem setul de antrenare de 1000 de ori

#creem datasetul care ne returneaza x_train si y_train
dataset = ChatDataset()
#pyTorch amesteca datele si le pune in batch-uri de cate 8
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#device ne zice daca antrenam pe gpu sau cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#modelul este creat cu clasa neuralNet si mutat pe dispozitivul ales de device
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#crossentropy - cea mai folosita functie de pierdere pentru clasificare multi-clasa
#compara predictia modeului cu eticheta reala
criterion = nn.CrossEntropyLoss()
#Adam = adaptive moment estimation
#ajusteaza automat la fiecare pas cum modificam greutatile fiecarui neuron
#bazat pe informatia anterioara generata de model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    #luam pe rand din loader cate o pereche de antrenament de lista de bag of words
    #si label-uri
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        #datele trec prin retea si primim o predictie, apoi calculam precizia
        outputs = model(words)
        loss = criterion(outputs, labels)

        #resetam gradientul ca sa nu se adune pe parcurs, apoi cu loss.backward
        #canculam gradientul din informatiile anterioare si actualizam valorile
        #neuronilor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #afisam loss-ul la fiecare 100 de epoci de antrenare
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss={loss.item():.4f}')


#SALVAM MODELUL
#creem un dictionar care contine datele de care avem nevoie pentru a 
#reincarca modelul fara a-l antrena din nou
data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_words': all_words,
    'tags': tags,
}

FILE = 'data.pth'
#salvam dictionarul
torch.save(data, FILE)
print(f'We have {len(tags)} topics !')
print(f'Training complete. File saved to {FILE}')

# with open('intents.json', 'r') as f:
#     intents = json.load(f)

# data = torch.load(FILE)
# input_size = data['input_size']
# hidden_size = data['hidden_size']
# output_size = data['output_size']
# all_words = data['all_words']
# tags = data['tags']
# model_state = data['model_state']

model_state = model.state_dict()

#reconstuim modelul cu greutatile antrenate
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
#il punem in modul de predictie, nu de antrenare
model.eval()

bot_name = 'Nicusor'
print("Let's chat! Type 'quit' to exit.")

#incepem interactiunea cu botul intr-un loop infinit
while True:
    #luam inputul intr-o variabila
    sentence = input('You: ')
    #daca scriem quit se iese din program
    if sentence.lower() == 'quit':
        break

    #tokenizam inputul, creem bag of words
    tokens = tokenize(sentence)
    bow = bag_of_words(tokens, all_words)
    bow = bow.reshape(1, bow.shape[0])
    #trannfsormam bag of words intrun tensor si il trimitem pe device-ul ales
    bow_tensor = torch.from_numpy(bow).to(device)

    #modelul face o predictie dupa ce a primit tensorul
    #in output se afla un vector de scoruri pentru fiecare tag
    output = model(bow_tensor)
    #se alege scorul cel mai mare ca fiind predictia corecta
    _, predicted = torch.max(output, dim=1)
    #alegem tagul caruia ii apartine scorul maxim si ii convertim indexul in denumire
    tag = tags[predicted.item()]

    #cu softmax transformam scorurile in probabilitati
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()] #probabilitatea tagului ales

    #daca modelul este suficient de sigur si ne da o probabilitate mai mare de 75%
    #alegem random un raspuns din raspunsurile posibile pentru acel tag
    if prob.item() > 0.75:
        for intent in intents_data['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])
                print(f"{bot_name}: {response}")
    else:
        print("I do not understand...")