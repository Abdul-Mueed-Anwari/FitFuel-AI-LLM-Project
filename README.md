# FitFuel Coach

FitFuel Coach builds a weekly workout and a day of meals for a beginner, then lets that person change the same week in chat.

Abdul Mueed Anwari, NLP and LLM bootcamp.

This is not medical advice and it does not replace a doctor or dietitian.

## Run it

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python -m streamlit run app.py
```

Open the local URL Streamlit prints. Fill in the form and press **Build my week**. The week is saved under that name in the sidebar.

Plans are stored in `data/fitfuel.db` on this machine. That file is not part of the repo.

## Language model

The week itself comes from the local exercise and meal lists, so the app runs with no API key. With `GROQ_API_KEY`, every chat message is sent to Groq along with the profile and the current week. The reply is a normal conversation. A requested change is built and checked by the tools, and the reply says whether it was added.

Put one of these in `.env`:

- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`

On Streamlit Community Cloud, put the same name in the app secrets instead of a file.

## Try it

1. Build a week for someone who wants fat loss, 4 days, a home setup, and no dairy. Ayesha in `evaluate.py` is that example.
2. Ask `Make dinners under 500 calories`. Dinner should be 500 kcal or under.
3. Ask `What if I only have 30 minutes?` Each day should have 3 exercises.
4. Ask `Make my meals vegetarian.` Meat and fish should leave the meals.
5. Ask `What is on my week?` The reply should talk about the saved week, not paste a web page.
6. Ask about a knee injury. The coach should refuse and leave the saved week unchanged.
7. Ask `Add chicken haleem for dinner`. If it passes the check it goes on the week. Ask `Add yogurt for breakfast` for a dairy-free person and the coach should say it was not added.
8. Press **Start a new week**. The same person should get a different week.

## Sample check

```powershell
.\.venv\Scripts\python evaluate.py
.\.venv\Scripts\python -m unittest tests.test_coach tests.test_tools
```

`evaluate.py` builds plans for the 10 sample people, including Ayesha's dinner cap and Joy's vegetarian follow-up.

## How the agents work

| Agent | Job | Tools |
|---|---|---|
| Coach | Reads the profile, routes the message, saves the plan, writes the reply | `get_user_profile`, `save_plan`, `update_plan` |
| Workout | Builds or swaps the weekly sessions | `knowledge_retriever` |
| Nutrition | Builds meals and the calorie estimate | `knowledge_retriever`, `calculator` |
| Validator | Blocks diet, allergy, equipment, and length problems | `validate_plan` |

Questions can also use Wikipedia and DuckDuckGo. Those results stay in How this was checked. A new exercise or meal is checked first, then added, or refused with the reason. The graph is LangGraph in `fitfuel/graph.py`.

## Deploy

The app starts from `app.py`.

1. Push this folder to GitHub. Do not commit `.env` or `data/*.db`.
2. Create an app on [Streamlit Community Cloud](https://streamlit.io/cloud). Main file: `app.py`. Requirements file: `requirements.txt`.
3. Add a free model key under Secrets if you want model-written replies.

On Streamlit Community Cloud the SQLite file is shared by everyone using that app, and it can be wiped when the app restarts. Saved people on your laptop stay in `data/fitfuel.db`.

## Safety

- Injury, diagnosis, pregnancy, and treatment requests get a refusal and do not change the plan.
- Daily diets under 1400 kcal are refused. A single meal can still be capped, such as dinners under 500 kcal.
- Targets stay in a moderate range. Meal numbers are estimates, not lab values.
