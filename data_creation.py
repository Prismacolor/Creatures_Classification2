import random
import csv
import os

base_path = os.getcwd()
data_path = base_path + r'\data_samples\\'
master_data_list = []


# functions to create data objects
def create_wyvern():
    max_count = 1812
    x = 0
    wings = [2, 4]

    while x <= max_count:
        wyvern = {}
        wyvern['Legs'] = 2
        wyvern['Wings'] = random.choice(wings)
        wyvern['Length'] = random.randrange(190, 275)
        wyvern['Height'] = random.randrange(125, 175)
        wyvern['Environment'] = 'volcanoes'
        wyvern['Classification'] = 'Wyvern'

        master_data_list.append(wyvern)
        x += 1


def create_serpent():
    max_count = 2034
    x = 0
    environment = ['air', 'water']

    while x <= max_count:
        serpent = {}
        serpent['Legs'] = 4
        serpent['Wings'] = 0
        serpent['Length'] = random.randrange(425, 650)
        serpent['Height'] = random.randrange(50, 75)
        serpent['Environment'] = random.choice(environment)
        serpent['Classification'] = 'Serpent'

        master_data_list.append(serpent)
        x += 1


def create_flying_serpent():
    max_count = 1618
    x = 0

    while x <= max_count:
        flying_serpent = {}
        flying_serpent['Legs'] = 4
        flying_serpent['Wings'] = 2
        flying_serpent['Length'] = random.randrange(450, 675)
        flying_serpent['Height'] = random.randrange(65, 85)
        flying_serpent['Environment'] = 'air'
        flying_serpent['Classification'] = 'Flying Serpent'

        master_data_list.append(flying_serpent)
        x += 1


def create_dragon():
    max_count = 2575
    x = 0
    wings = [2, 4, 6]
    environment = ['mountain', 'volcanoes', 'cave']

    while x <= max_count:
        dragon = {}
        dragon['Legs'] = 4
        dragon['Wings'] = random.choice(wings)
        dragon['Length'] = random.randrange(325, 450)
        dragon['Height'] = random.randrange(200, 275)
        dragon['Environment'] = random.choice(environment)
        dragon['Classification'] = 'Dragon'

        master_data_list.append(dragon)
        x += 1


def create_drake():
    max_count = 2302
    x = 0
    legs = [2, 4]
    environment = ['desert', 'volcanoes', 'air', 'woods', 'mountain', 'cave']

    while x <= max_count:
        drake = {}
        drake['Legs'] = random.choice(legs)
        drake['Wings'] = 2
        drake['Length'] = random.randrange(100, 125)
        drake['Height'] = random.randrange(50, 75)
        drake['Environment'] = random.choice(environment)
        drake['Classification'] = 'Drake'

        master_data_list.append(drake)
        x += 1


def create_wyrm():
    max_count = 2021
    x = 0
    environment = ['cave', 'swamp']

    while x <= max_count:
        wyrm = {}
        wyrm['Legs'] = 0
        wyrm['Wings'] = 0
        wyrm['Length'] = random.randrange(450, 650)
        wyrm['Height'] = random.randrange(35, 50)
        wyrm['Environment'] = random.choice(environment)
        wyrm['Classification'] = 'Wyrm'

        master_data_list.append(wyrm)
        x += 1


# function to assign numbers to each creature
def add_numbers(some_list):
    creature_count = 0
    for item in some_list:
        creature_count += 1
        item['Number'] = creature_count


# function to save data to a csv
def create_excel(some_file, some_list):
    with open(data_path + some_file, 'w+', newline='') as f:
        writer = csv.DictWriter(f, ['Number', 'Legs', 'Wings', 'Length', 'Height', 'Environment', 'Classification'])
        writer.writeheader()

        for item in some_list:
            writer.writerow(item)


# calling all functions
create_wyvern()
create_serpent()
create_flying_serpent()
create_dragon()
create_drake()
create_wyrm()

# shuffle the data and split it into a training and test set
random.shuffle(master_data_list)
test_list = master_data_list[-50:]
test_list2 = master_data_list[-35:]
del master_data_list[-50:]

add_numbers(master_data_list)
add_numbers(test_list)
add_numbers(test_list2)

# store the data in spreadsheets
create_excel('creatures_data_set.csv', master_data_list)
create_excel('creatures_test_set.csv', test_list)
create_excel('creatures_test_set2.csv', test_list2)

