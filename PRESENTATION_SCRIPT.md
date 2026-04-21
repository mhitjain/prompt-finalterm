# Presentation Script — Dewey RL Adaptive Tutoring

---

## Commands — Run These Before You Start

```bash
# Terminal 1
streamlit run compare_app.py
# → http://localhost:8501

# Terminal 2
uvicorn api_server:app --reload --port 8000

# Terminal 3
streamlit run api_ui.py --server.port 8503
# → http://localhost:8503
```

**Open in browser tabs before recording:**
1. `http://localhost:8501` — main demo
2. `http://localhost:8503` — API demo
3. `http://localhost:8000/docs` — Swagger
4. `results/figures/policy_heatmap.png` — open in Preview
5. `results/figures/full_comparison.png` — open in Preview

---

## PART 1 — Opening (45 sec)

> "So in 1984, a researcher named Benjamin Bloom found something remarkable.
> Students who got one-on-one human tutoring performed two standard deviations better
> than students in a regular classroom. That's basically the difference between
> being average and being in the top 2%.
>
> The problem is we can't give every student a personal tutor. There aren't enough teachers.
>
> So the question this project tries to answer is — can we use reinforcement learning
> to build something that actually adapts to each student?
> Not just follow a script, but genuinely learn how to teach.
>
> That's what this is."

---

## PART 2 — Architecture (60 sec)

*Point to architecture diagram*

> "Here's how it works. Three layers.
>
> At the top, the RL decision engine. Two algorithms running together —
> PPO picks from 11 teaching actions: explain, ask questions at different difficulties,
> give hints, switch topics, encourage. And two Thompson Sampling bandits handle
> finer decisions — one for question difficulty, one for content style.
>
> I used two separate algorithms because PPO is good at long-horizon planning,
> thinking about what happens 10 steps from now. Thompson Sampling is better
> at quick exploration — should I try an example or a hint right now?
> They're actually complementary, and I'll show the idea that proves that.
>
> Below that are the Dewey agents — Ada for Calculus, Newton for Physics, Grace for Algorithms.
> These take the RL's decision and generate actual tutoring content using an LLM.
> I used Groq right now but it works with OpenAI, Anthropic, or pure simulation with no key.
>
> And there are three custom tools the agents use — a Knowledge Graph for optimal topic ordering,
> a Difficulty Estimator using Item Response Theory that computes each student's
> Zone of Proximal Development, and a Performance Tracker that logs everything.
>
> That IRT model — this is worth saying — is the exact same statistical framework
> Duolingo uses to estimate how hard a question is for a specific user.
> So our student simulator isn't just made-up — it's built on the same foundation
> as one of the largest adaptive learning platforms in the world.
>
> The whole thing is wrapped in a FastAPI server so any LMS or app can plug in over HTTP."

---

## PART 3 — Live Demo (3 min)

*Go to http://localhost:8501*

> "Okay this is the main demo. I'm running three systems on the exact same student —
> same starting knowledge, same profile — and we watch them diverge.
> SLOW_LEARNER, 40 steps, 20k training steps."

*Click Start, while training:*

> "So right now it's training the RL agent — 20,000 simulated teaching sessions,
> no API calls at all. The LLM only gets called during actual tutoring, not training.
> That's what makes this practical — you don't spend anything training."

*When columns appear:*

> "Okay so look at the three columns.
>
> Fixed Script — EXPLAIN, ASK_MEDIUM, HINT, repeat forever.
> It literally doesn't know if you're confused, bored, or already know this.
>
> Untrained RL — just random actions, no learning. Sometimes helpful, mostly chaotic.
>
> Trained RL — watch what it actually does. When the student struggles, it backs off.
> When engagement is high, it pushes harder. When a topic's done, it moves on.
> Nobody programmed those rules. It figured them out."

*Scroll to scorecard:*

> "Fixed Script used 3 actions — same three on repeat.
> The trained RL is using 7 or 8 different ones because it's actually responding
> to what's happening with the student."

*Scroll to 150-step chart:*

> "This is the most important chart. At 50 steps, honestly, all three look similar.
> But at 150 steps, Fixed Script engagement drops to 0.16 — the student's basically given up.
>
> Trained RL holds at 0.64, gets 34% more knowledge gain, and 746% more cumulative reward.
>
> And the reason makes sense — once a student disengages, they stop learning.
> The RL learned to prevent that. Fixed Script never does."

---

## PART 4 — What Did It Actually Learn? (90 sec)

*Open policy_heatmap.png*

> "So this is where it gets interesting. We can look inside the policy and see what it learned.
>
> The left chart plots every single step from 60 real tutoring sessions.
> Each dot is one step — x-axis is the student's knowledge, y-axis is their engagement.
> Color tells you what action the RL took.
>
> See that red dashed line at the bottom? That's the disengagement threshold —
> below that, the student is about to give up.
> Look at what's clustered right near that line — those cyan dots are ENCOURAGE.
> The policy learned to encourage students specifically when they're close to disengaging.
> That's not a rule anyone wrote. It emerged from the reward signal.
>
> The bar chart on the right shows the overall action distribution —
> SHOW_EXAMPLE dominates, which makes sense, examples drive the most knowledge gain.
> But it's using ASK_MEDIUM, ASK_HARD, EXPLAIN, ENCOURAGE regularly —
> genuinely using the full action space based on what the student needs."

*Open full_comparison.png*

> "And here's the comparison against five real algorithms from the educational technology
> literature — not toy baselines, actual ITS methods that schools use.
> Random, Fixed Script, Mastery Learning, Zone of Proximal Development, Spaced Repetition.
>
> We beat all five with statistical significance.
> Against ZPD and Spaced Repetition, Cohen's d is 2.2 — that's a massive effect size,
> anything above 0.8 is considered large.
>
> The interesting finding is that ZPD and Spaced Repetition actually do worse than
> Fixed Script. The reason — they optimize each step locally, no long-horizon planning.
> PPO has a value function that thinks about future rewards,
> so it beats methods that only look at the current moment."

---

## PART 5 — Production API (75 sec)

*Go to http://localhost:8503*

> "Now the deployment story. This entire page has zero RL code.
> It's just a frontend making HTTP calls to our API.
> This is exactly what a university LMS would look like."

*Create session, click Take Next Step 3-4 times*

> "Every step returns structured JSON — action, content, knowledge gain, engagement, reward.
> The application doesn't need to understand RL at all.
> It just calls step and renders what comes back.
>
> The server handles multiple sessions at once, background training, health checks, all of it."

*Switch to http://localhost:8000/docs*

> "Full Swagger documentation. This isn't a research prototype — it's actually deployable."

---

## PART 6 — Technical Details (40 sec)

> "A few quick things worth mentioning.
>
> The reward function has a sparse plus-10 bonus when a topic hits 85% mastery.
> That one design choice is why the RL sticks with a topic instead of jumping around —
> it's actually chasing those mastery bonuses.
>
> The Thompson Sampling bandits start with LinUCB for the first 20 pulls
> before switching to Thompson Sampling proper — that prevents cold-start failure
> where it gets stuck on one arm before seeing enough data.
>
> And the student simulator uses Item Response Theory — same model behind standardized
> tests like the GRE, and the same framework Duolingo uses for adaptive difficulty.
> So when I say the simulator is realistic,it's not just random noise,
> it's the industry-standard way of modeling how a specific student responds
> to a specific question at a specific difficulty level."

---

## PART 7 — Close (25 sec)

> "So to wrap up — this is a tutoring system that learns how to teach
> through reinforcement learning, beats five established educational algorithms,
> and is deployed as a production API any platform can use today.
>
> The core result is that 150-step chart — Fixed Script destroys engagement over time,
> RL sustains it, and that difference compounds into 34% more learning.
>
> That's the RL answer to Bloom's Two-Sigma Problem. Thanks."

---

## If They Ask...

**"Isn't this just ChatGPT with a good prompt?"**

> "You could try to approximate it with a prompt, sure. But you'd have to already know
> the optimal strategy to write the prompt — the RL discovered it from data.
> Nobody told it to encourage students near the disengagement threshold, it figured that out.
>
> Also, a prompt needs an LLM call for every single decision.
> Our RL policy runs on a 26-number state vector in microseconds.
> That's roughly 100 times cheaper per step."

**"Why does it only look better at 150 steps, not 50?"**

> "At 50 steps engagement is still high across all systems, so the difference
> hasn't had time to compound. By 150 steps, untrained RL has been randomly
> throwing hard questions at the student, engagement has drifted down,
> and knowledge gains are slowing. Trained RL kept engagement higher for longer —
> that sustained engagement is what drives the gap."

**"What would you build next?"**

> "Three things. Replace the hard-coded mode switching between LEARNING, ASSESSMENT,
> and CONTENT with a learned meta-policy — that's still a hand-written rule right now.
> Second, test on real students with an A/B trial.
> Third, learn the reward function from teacher demonstrations using inverse RL
> instead of hand-tuning the weights."

**"What are the limitations?"**

> "The simulator is synthetic, real students are messier than IRT models.
> The policy was trained on 50-step episodes so it's slightly out of distribution
> beyond step 50 — that's why engagement still drifts down a bit even for trained RL.
> And the reward weights are hand-tuned, learned weights would be stronger."
