import random

SUGGESTIONS = {
    "Happy": [
        "Keep smiling! Enjoy your day.",
        "Celebrate your wins, big or small!",
        "Share your happiness with a friend.",
        "Spread those good vibes—do something kind for someone!",
        "You're glowing—keep doing what you're doing!",
        "Dance like nobody’s watching—even if someone is!",
        "Write a thank-you note to someone who made you smile.",
        "Try telling someone a silly joke and see if it’s contagious!",
        "Make yourself an ice cream sundae—just because."
    ],
    "Sad": [
        "It's okay to feel sad; allow yourself the time to feel.",
        "Reach out to someone who cares. You're not alone.",
        "Watch your favorite feel-good movie or show.",
        "Go for a walk, even a short one. Nature helps.",
        "Write down your thoughts—it’s more powerful than you think.",
        "Call a friend and chat about old happy memories.",
        "Treat yourself with your favorite snack or drink.",
        "Listen to uplifting music—bonus points for singing along.",
        "Write about something you’re grateful for today."
    ],
    "Angry": [
        "Take a few deep breaths to cool down.",
        "Channel your anger into something productive—like exercise!",
        "Try listening to calming music.",
        "Journaling your thoughts can help you gain clarity.",
        "Take a 5-minute break before responding—you’ve got this.",
        "Find a funny video and laugh the anger away.",
        "Take a brisk walk or do a quick workout to let off some steam.",
        "Squeeze a stress ball or punch a pillow—safely!",
        "Try drawing a silly cartoon about what made you angry."
    ],
    "Neutral": [
        "Why not try something new today? A small adventure might help!",
        "Take a moment to enjoy something simple—a walk, a song, a tea.",
        "You’re steady and focused—use this moment to plan ahead.",
        "Find a fun hobby or creative task to mix up your routine.",
        "Could be a great day to start a new book or podcast.",
        "Try a five-minute meditation to recharge your batteries.",
        "Step outside and notice five new things—challenge accepted!",
        "Play a quick online game or brain teaser to mix things up.",
        "Doodle something random for a quick creative break."
    ],
    "Fear": [
        "Try slow, deep breaths to calm your nerves.",
        "Courage isn't the absence of fear—you're already on your way!",
        "Take control by tackling one small thing at a time.",
        "Visualize a peaceful place for 30 seconds, eyes closed.",
        "Face your fear in tiny steps—you’re stronger than you think.",
        "Call a friend and share what’s making you anxious—it helps.",
        "Make a calming tea or hot chocolate and sip slowly.",
        "Do a “power pose” in the mirror for 2 minutes.",
        "Watch a silly video of dancing cats or puppies."
    ],
    "Disgust": [
        "Take a break and reset your senses.",
        "Do something positive that makes you feel clean and refreshed.",
        "Change your environment—it helps reset your mood.",
        "Focus on what makes you feel good and healthy.",
        "Smile at something silly. It works better than you'd think!",
        "Open a window and take a few fresh breaths.",
        "Put on your favorite scent or hand lotion.",
        "Clean or organize a small area around you.",
        "Watch a short, oddly satisfying cleaning video."
    ],
    "Surprise": [
        "Take a moment and let it sink in—surprises can be gifts!",
        "React slowly—sometimes surprises lead to amazing outcomes.",
        "Stay open-minded. Big changes often come in surprise boxes!",
        "Talk to someone about how you’re feeling; surprises are meant to be shared.",
        "Good or bad, surprises keep life interesting—embrace it!",
        "Jot down three unexpected things that happened lately—good or bad!",
        "Celebrate the surprise with a little victory dance.",
        "Call someone and share your surprise moment.",
        "Try something you’ve never done before, no matter how small."
    ]
}

JOKES = {
    "Happy": [
        "Why did the tomato turn red? Because it saw the salad dressing!",
        "What do you call fake spaghetti? An impasta!",
        "Why don’t scientists trust atoms? Because they make up everything!",
        "How does a penguin build its house? Igloos it together!",
        "Why don’t skeletons ever go trick or treating? Because they have no body to go with!",
        "Did you hear about the claustrophobic astronaut? He just needed a little space!",
        "Why was the math lecture so long? The professor kept going off on a tangent!"
    ],
    "Sad": [
        "Why did the cookie go to the doctor? Because it felt crummy!",
        "Why was the broom late? It swept in!",
        "What’s a sad emoji’s favorite candy? Blue raspberry!",
        "How do you comfort a JavaScript bug? You console it.",
        "Why are ghosts so bad at lying? Because you can see right through them.",
        "Why did the golfer bring two pants? In case he got a hole in one.",
        "Why did the teddy bear say no to dessert? He was stuffed!"
    ],
    "Angry": [
        "Why did the angry computer go to therapy? It had too many bytes!",
        "What’s the angry cat’s favorite color? Rrrrrred!",
        "Why was the math book angry? Because it had too many problems!",
        "Why did the angry music conductor get arrested? For assault and battery!",
        "Why do shoes always get angry? They have too many heels!",
        "Did you hear about the guy who lost his left side? He's all right now.",
        "Why did the storm get so mad? It had a lot of thunder feelings!"
    ],
    "Neutral": [
        "Why did the bicycle fall over? Because it was two tired!",
        "Why do bees have sticky hair? Because they use honeycombs!",
        "What did one wall say to the other wall? I’ll meet you at the corner.",
        "Why was the math book so neutral? Because it had too many problems.",
        "Why did the computer take a nap? It needed to reboot its mood.",
        "How do you find Will Smith in the snow? Look for fresh prints.",
        "Why did the scarecrow get promoted? He was outstanding in his field!"
    ],
    "Fear": [
        "Why don’t skeletons fight each other? They don’t have the guts!",
        "Why was the student afraid of the math test? Because it was full of problems.",
        "Why did the ghost go to therapy? To get over his boos!",
        "Why did the skeleton stay home from the party? Because he had no guts!",
        "How do monsters like their eggs? Terri-fried.",
        "Why did the vampire read the newspaper? He heard it had great circulation.",
        "Why don’t mummies take vacations? They’re afraid to unwind!"
    ],
    "Disgust": [
        "Why did the salad blush? Because it saw the dressing!",
        "What do you call cheese that isn’t yours? Nacho cheese!",
        "Why don’t eggs tell jokes? They’d crack each other up!",
        "Why did the toilet paper roll down the hill? To get to the bottom.",
        "Why don’t oysters share their pearls? Because they’re a bit shellfish!",
        "Why did the banana go to the doctor? Because it wasn’t peeling well.",
        "Why did the frog take a bath? It was a little jumpy."
    ],
    "Surprise": [
        "How do you organize a space party? You planet!",
        "What did the zero say to the eight? Nice belt!",
        "Why did the chicken go to the séance? To talk to the other side!",
        "Did you hear about the kidnapping at the playground? Don’t worry, he woke up!",
        "Why was the calendar so popular? It had a lot of dates!",
        "Why did the student eat his homework? His teacher said it was a piece of cake.",
        "Did you hear about the cheese factory explosion? There was nothing left but de-brie."
    ]
}

def get_suggestion(emotion):
    """Return 1 random suggestion for the given emotion."""
    return random.choice(SUGGESTIONS.get(emotion, ["Just breathe. You're doing great!"]))

def get_joke(emotion):
    """Return 1 random joke for the given emotion."""
    return random.choice(JOKES.get(emotion, ["Why did the scarecrow win an award? Because he was outstanding in his field!"]))
