# Question auto generate for trigger extracted, subject extracted, object extracted, time extracted, place extracted
import random

def TRIGGER_GENERATE(trigger1=None,subject1=None,object1=None,time1=None,place1=None):
    PATTERN_1 = ["What is the main action in the text ?",
                # "The action ?",
                "What action is central to the narrative ?"]

    PATTERN_2 = []
    if trigger1:
        PATTERN_2.append(f"Except {subject1} {trigger1}, what action happened ?")
        PATTERN_2.append(f"Besides {subject1} {trigger1} {object1}, what else occurred ?")
        return random.choice(PATTERN_2).replace(" None","")
    else:
        return random.choice(PATTERN_1)

def SUBJECT_GENERATE(trigger1,subject1=None,object1=None,time1=None,place1=None):
    PATTERN = []
    # PATTERN.append(f"What {trigger1} happened?")
    # PATTERN.append(f"Who or what {trigger1}?")
    # PATTERN.append(f"What {trigger1} affected?")
    PATTERN.append(f"Who or what was involved in {trigger1} ?")
    # PATTERN.append(f"Which entity was primarily responsible for the {trigger1}?")
    return random.choice(PATTERN)

def OBJECT_GENERATE(trigger1,subject1=None,object1=None,time1=None,place1=None):
    PATTERN = []
    # PATTERN.append(f"Whose {trigger1} by?")
    # PATTERN.append(f"What Object {trigger1}?")
    PATTERN.append(f"What was {trigger1} affected ?")
    return random.choice(PATTERN)

def TIME_GENERATE(trigger1, subject1=None, object1=None, time1=None, place1=None):
    PATTERN = []
    PATTERN.append(f"When did the {trigger1} happen ?")
    # PATTERN.append(f"At what time did the {trigger1} occur?")
    # PATTERN.append(f"When did the {trigger1} take place in the story?")
    # Add more patterns as needed
    return random.choice(PATTERN)

def PLACE_GENERATE(trigger1, subject1=None, object1=None, time1=None, place1=None):
    PATTERN = []
    PATTERN.append(f"Where did the {trigger1} take place ?")
    # PATTERN.append(f"In which location did the {trigger1} occur?")
    # PATTERN.append(f"What is the setting of the {trigger1}?")
    # Add more patterns as needed
    return random.choice(PATTERN)

if __name__ == "__main__":
    # Example usage
    trigger = "attack"
    subject = "the enemy"
    object_ = "the castle"
    time_ = "yesterday"
    place_ = "on the battlefield"
    print(TRIGGER_GENERATE(trigger, subject, object_))
    print(SUBJECT_GENERATE(trigger))
    print(OBJECT_GENERATE(trigger))
    print(TIME_GENERATE(trigger))
    print(PLACE_GENERATE(trigger))