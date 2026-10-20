#!/usr/bin/env python3
"""CS-45 authored text inventory. The '|' marks the cut point (it is stripped before synthesis).

Pair families (all pairs splice ONE rendered user recording; see tt_build.py):
  CUT_IC    : same recording, cut mid-clause (label incomplete) vs run to its end (label complete)
  CUT_WAIT  : same recording, cut after a self-contained hold request (label wait) vs the full
              utterance whose continuation changes the intent (label complete)
  CTX_BC    : same short acknowledgement waveform; only the PRECEDING ASSISTANT CONTEXT differs
              (assistant just asked a question and stopped -> complete; assistant mid-explanation and
              still talking -> backchannel)
"""

# Every prefix before '|' ends on a word that cannot close a clause (determiner, preposition,
# conjunction, auxiliary, complementiser, possessive). Unambiguous-incomplete cuts only; a prefix
# that could stand as a finished request is NOT admitted (human review checks this).
CUT_IC = [
    "I was thinking we could go to the | park after lunch tomorrow.",
    "Could you please send me the | report from last week?",
    "I would like to book a table for | two on Friday evening.",
    "My flight lands at six but I still need to | pick up my luggage first.",
    "Can you remind me to call my sister when | I get home tonight?",
    "We should probably leave early because the | traffic gets really bad around five.",
    "I think the best option is to wait until | the prices come down a little.",
    "Please turn the thermostat down to | sixty eight degrees before bed.",
    "She told me that the meeting was moved to | Thursday afternoon at three.",
    "I'm trying to find a recipe that uses | chicken and lemon but no dairy.",
    "Do you know if there is a pharmacy near | the train station that stays open late?",
    "If you have time this weekend I would love to | show you the new garden.",
    "The package should arrive sometime between | Monday and Wednesday of next week.",
    "I've been meaning to ask you about the | conference in Berlin next spring.",
    "Let's set up a call with the whole team and | go over the budget together.",
    "He said he would bring the camera and also | a spare battery for the long hike.",
    "Could you add milk and eggs and maybe some | fresh bread to my shopping list?",
    "I don't really remember where I put my | keys after I came back from the store.",
    "What time does the museum open on | Sunday morning during the winter months?",
    "I need to renew my passport before the | end of the summer because we are traveling.",
    "Tell me how to get from the airport to | the hotel without taking a taxi.",
    "My daughter wants a birthday cake that looks like a | big purple dinosaur with sprinkles.",
    "I'm not sure whether I should take the job in | Chicago or stay where I am.",
    "Can you read me the last message from | my brother about the dinner plans?",
    "We are looking for an apartment with two bedrooms and a | balcony that faces the river.",
    "The reason I called is that I wanted to see if | you could cover my shift on Saturday.",
    "Set an alarm for seven thirty tomorrow and | another one ten minutes later.",
    "I read an article yesterday about how the | ocean is getting warmer every single year.",
    "Would you mind helping me write an email to my | landlord about the broken heater?",
    "It's been raining all morning so I decided to | stay inside and finish my book.",
    "Please play something relaxing while I | cook dinner for the kids.",
    "The doctor said I should avoid caffeine for at | least two weeks after the surgery.",
    "I wonder if there is any chance that the | concert will be rescheduled to next month.",
    "Ask the front desk to send up some extra | towels and a bottle of water.",
    "Our neighbors are having a party on Saturday and they invited | us to come over around eight.",
    "I can't decide between the red jacket and the | blue one for the wedding.",
    "Could you check whether my order has shipped yet or if it's | still waiting at the warehouse?",
    "He has been working on this project since | last October without any real break.",
    "Once the oven is hot enough I'll put in the | tray of vegetables and set a timer.",
    "My phone keeps telling me that the storage is | almost full but I deleted everything.",
    "What I really want to know is how much it would cost to | repair the old bicycle.",
    "Translate the sentence where the cat sits on the | windowsill into Spanish for me.",
    "She was surprised that nobody had told her about the | change in the schedule.",
    "Let me know when the train to Boston is about to | leave the platform.",
    "I just got back from a long trip and my suitcase is | still sitting by the front door.",
    "Do you think we should invite everyone from the old office or just | the people we still talk to?",
    "The kids have a soccer game on Saturday morning, so we need to | be at the field by nine.",
    "Remind me that I have to pick up the dry cleaning before the | store closes this evening.",
    "I would love some advice on how to prepare for an | interview at a small software company.",
    "There was a time when I could run five miles without | stopping but those days are gone.",
    "Can you find out whether the library is open on | public holidays or only on weekdays?",
    "I'm going to need a ride to the airport on | the fifteenth around five in the morning.",
    "Write down that the plumber is coming on Tuesday between | eight and noon to fix the sink.",
    "They announced that the new bridge will be finished by the | end of next year at the earliest.",
    "My grandmother used to make a soup with potatoes and | leeks that I have never forgotten.",
    "Is it going to rain this weekend or will it be | sunny enough for a picnic at the lake?",
    "We booked the cabin for three nights but we might want to | stay longer if the weather is nice.",
    "I keep forgetting to water the plants that are on the | shelf next to the kitchen window.",
    "The instructions say to mix the flour with the | butter until the dough feels crumbly.",
    "After the meeting I have to call the bank and then | go straight to the dentist.",
    "Honestly I think the movie was good, although the ending felt a | little bit rushed to me.",
    "Could you summarize the main points of the lecture that I | missed on Monday morning?",
    "Please put a reminder on my calendar for the day before | my mother's birthday party.",
    "I heard there is a new coffee shop on the corner of | Main Street and Fifth Avenue.",
    "He asked whether I had ever been to the mountains in | the north during the winter.",
    "Can you tell me what the weather will be like in Lisbon when | I land on Friday afternoon?",
]

# Hold requests: the prefix before '|' is a self-contained request to wait (label wait); the full
# utterance continues with a content change (label complete).
CUT_WAIT = [
    "Hold on a second, | that is not what I asked for at all.",
    "Wait a moment, | I think you have the wrong address.",
    "Hang on, | that sounds like the wrong date to me.",
    "Give me a second, | the number you read out was incorrect.",
    "One moment please, | I want to change my order before you send it.",
    "Just a minute, | that isn't the song I meant, play the other one.",
    "Hold on, | my husband just walked in and he disagrees.",
    "Wait, | stop there because I already have a ticket for that.",
    "Give me a moment, | you misunderstood and I need to explain it again.",
    "Hang on a second, | I forgot to tell you the room is for three people.",
    "Just a second, | that price looks far too high for a single night.",
    "Hold on for a moment, | the other name is spelled with two L's.",
    "Wait a second, | I never said anything about moving the meeting.",
    "One second, | that doesn't match what the doctor told me.",
]

# Short acknowledgements for the context-contrast pairs (one source recording per pair).
CTX_ACKS = [
    "Yeah.", "Okay.", "Right.", "I see.", "Sure.",
    "Got it.", "Yes.", "Mm hm.", "Oh, okay.", "Alright.",
]
CTX_REPEATS = 3  # pairs per acknowledgement phrase -> 30 pairs

# Assistant turn that ENDS with a question and then stops (the user's acknowledgement is an answer).
ASSIST_QUESTION = [
    "Would you like me to book that table for Friday evening?",
    "Do you want me to send the summary to your email?",
    "Should I set the reminder for eight in the morning?",
    "Is that the address you want me to use for the delivery?",
    "Shall I go ahead and cancel the second appointment?",
    "Do you want the short version or the full explanation?",
    "Would it help if I read the instructions out loud?",
    "Should I add those three items to your shopping list?",
    "Are you ready for me to start the timer now?",
    "Do you want me to keep going with the next step?",
    "Would you like me to call the restaurant for you?",
    "Is it okay if I turn the lights off in the living room?",
]

# Long assistant explanations that keep going while the user backchannels mid-way.
# (>= 9 s of speech so that an acknowledgement at ~3 s is followed by >= 2.5 s of continuing speech.)
ASSIST_MONOLOGUE = [
    "So the way this works is that the system first looks at your calendar, then it finds a few open slots that fit the meeting, and after that it sends invitations to everyone on the list with a short note explaining the agenda.",
    "To make the soup you start by chopping the onions and the carrots, then you cook them slowly in a little olive oil until they are soft, and only after that do you add the stock and let everything simmer for about twenty minutes.",
    "The route I would suggest goes along the river for the first few miles, then it turns inland at the old stone bridge, and from there it is a gentle climb through the forest until you reach the village at the top of the hill.",
    "What happened with your order is that it was packed on Monday, handed to the carrier on Tuesday, and then held up at the regional depot because of the storm, but the latest update says it should be out for delivery tomorrow morning.",
    "When you set up the new router, the first thing to do is plug it into the modem, wait for the light to turn solid green, and then connect your phone to the temporary network so that you can choose a name and a password.",
    "The reason prices went up this quarter is a combination of higher shipping costs and a shortage of a few key components, and the company expects things to settle down once the new factory comes online later in the year.",
    "In the first part of the lecture the professor explains how the early settlers built their houses, then she moves on to the trade routes along the coast, and finally she compares those routes with the railways that came much later.",
    "Your training plan for the next four weeks starts with short easy runs on Monday and Wednesday, adds a longer run on Saturday, and then gradually increases the distance by about ten percent each week so your legs can adapt.",
    "If you decide to take the job, the benefits start after sixty days, which includes health coverage and a retirement plan, and the company also offers a small budget each year for books, courses, and conference travel.",
    "I checked three different places for you, and the cheapest one is across town but has parking, the second one is closer but charges extra for weekends, and the third one is the most expensive but includes breakfast and a late checkout.",
]

ASSIST_PROMPT = [
    "Sure, what can I help you with?",
    "Okay, go ahead, I'm listening.",
    "Alright, what would you like to do?",
    "Tell me what you need.",
    "Go ahead, I'm ready.",
    "What can I do for you today?",
]
