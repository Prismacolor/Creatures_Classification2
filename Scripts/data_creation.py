import random

master_data_list = []


def create_wyvern():
    max_count = 452
    x = 0
    wyvern_count = 0
    wings = [2, 4]

    while x <= max_count:
        wyvern_count += 1

        wyvern = {}
        wyvern['Number'] = wyvern_count
        wyvern['Legs'] = 2
        wyvern['Wings'] = random.choice(wings)
        wyvern['Length'] = random.randrange(115, 150)
        wyvern['Height'] = random.randrange(175, 225)
        wyvern['Environment'] = 'volcanoes'
        wyvern['Classification'] = 'Wyvern'

        master_data_list.append(wyvern)
        x += 1


def create_serpent():
    max_count = 417
    x = 0
    serpent_count = 0
    legs = [0, 4]
    environment = ['air', 'water', 'woods', 'swamp']

    while x <= max_count:
        serpent_count += 1

        serpent = {}
        serpent['Number'] = serpent_count
        serpent['Legs'] = random.choice(legs)
        serpent['Wings'] = 0
        serpent['Length'] = random.randrange(215, 250)
        serpent['Height'] = random.randrange(50, 80)
        serpent['Environment'] = random.choice(environment)
        serpent['Classification'] = 'Serpent'

        master_data_list.append(serpent)
        x += 1


def create_flying_serpent():
    max_count = 523
    x = 0
    flying_serpent_count = 0
    wings = [2, 4]

    while x <= max_count:
        flying_serpent_count += 1

        flying_serpent = {}
        flying_serpent['Number'] = flying_serpent_count
        flying_serpent['Legs'] = 0
        flying_serpent['Wings'] = random.choice(wings)
        flying_serpent['Length'] = random.randrange(250, 315)
        flying_serpent['Height'] = random.randrange(99, 125)
        flying_serpent['Environment'] = 'air'
        flying_serpent['Classification'] = 'Flying Serpent'

        master_data_list.append(flying_serpent)
        x += 1


def create_dragon():
    max_count = 362
    x = 0
    dragon_count = 0
    wings = [2, 4, 6]
    environment = ['mountain', 'volcanoes', 'cave']

    while x <= max_count:
        dragon_count += 1

        dragon = {}
        dragon['Number'] = dragon_count
        dragon['Legs'] = 4
        dragon['Wings'] = random.choice(wings)
        dragon['Length'] = random.randrange(125, 175)
        dragon['Height'] = random.randrange(65, 100)
        dragon['Environment'] = random.choice(environment)
        dragon['Classification'] = 'Dragon'

        master_data_list.append(dragon)
        x += 1


def create_drake():
    max_count = 425
    x = 0
    drake_count = 0
    legs = [2, 4]
    environment = ['desert', 'volcanoes', 'water']

    while x <= max_count:
        drake_count += 1

        drake = {}
        drake['Number'] = drake_count
        drake['Legs'] = random.choice(legs)
        drake['Wings'] = 2
        drake['Length'] = random.randrange(185, 225)
        drake['Height'] = random.randrange(50, 75)
        drake['Environment'] = random.choice(environment)
        drake['Classification'] = 'Drake'

        master_data_list.append(drake)
        x += 1


def create_wyrm():
    max_count = 513
    x = 0
    wyrm_count = 0
    environment = ['cave', 'swamp']

    while x <= max_count:
        wyrm_count += 1

        wyrm = {}
        wyrm['Number'] = wyrm_count
        wyrm['Legs'] = 4
        wyrm['Wings'] = 0
        wyrm['Length'] = random.randrange(295, 345)
        wyrm['Height'] = random.randrange(35, 50)
        wyrm['Environment'] = random.choice(environment)
        wyrm['Classification'] = 'Wyrm'

        master_data_list.append(wyrm)
        x += 1


create_wyvern()
create_serpent()
create_flying_serpent()
create_dragon()
create_drake()
create_wyrm()

print('hello')