"""Outfit Mate - Streamlit app.  Run with:  streamlit run app.py"""
from datetime import date, timedelta

import streamlit as st
from PIL import Image

from core import config, db, genstudio, llm, stylist
from core.vision import CATEGORIES, PALETTE, PATTERNS, describe, get_clip, save_image, tag_image

st.set_page_config(page_title="Outfit Mate", page_icon="👗", layout="wide")
db.init()


@st.cache_resource(show_spinner="Loading CLIP model (the first run downloads about 600 MB)...")
def clip_model():
    return get_clip()


ss = st.session_state
ss.setdefault("processed", set())
ss.setdefault("recs", None)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("👗 Outfit Mate")
    st.caption("Generative-AI personal stylist · Team T20")
    city = st.text_input("City (for weather)", config.DEFAULT_CITY)
    st.divider()
    st.markdown("**Status**")
    st.write(("🟢 LLM: " + config.LLM_PROVIDER + " / " + config.LLM_MODEL) if llm.enabled()
             else "🟡 LLM: off (rule-based fallback)")
    st.write("🟢 Weather API" if config.OPENWEATHER_API_KEY else "🟡 Weather: default 28 °C")
    st.write(f"👕 {len(db.list_garments())} garments in wardrobe")
    if llm.LAST_ERROR:
        st.warning(f"Last LLM error: {llm.LAST_ERROR[:200]}")

tab_w, tab_s, tab_p, tab_d, tab_i = st.tabs(
    ["👕 Wardrobe", "💬 Stylist", "📅 Week plan", "🎨 Design Studio", "📊 Insights"])

# ---------------------------------------------------------------- wardrobe
with tab_w:
    st.subheader("Digitise your wardrobe")
    st.caption("Tip: one garment per photo, laid flat or on a hanger, in good light.")
    files = st.file_uploader("Upload clothing photos", type=["jpg", "jpeg", "png", "webp"],
                             accept_multiple_files=True)
    if files and st.button("Process uploads", type="primary"):
        clip = clip_model()
        bar = st.progress(0.0)
        for i, f in enumerate(files):
            key = (f.name, f.size)
            if key in ss.processed:
                continue
            tags = tag_image(Image.open(f), clip)
            path = save_image(tags.pop("clean_image"))
            db.add_garment(tags, path)
            ss.processed.add(key)
            if tags["confidence"] < 0.35:
                st.warning(f"{f.name}: not sure it's a {tags['category']} — please check its tags below.")
            try:  # VAE reconstruction error (if the VAE is trained)
                chk = genstudio.photo_check(path)
                if chk and chk["unusual"]:
                    st.warning(f"{f.name}: this doesn't look like a typical garment photo "
                               f"(VAE anomaly z-score {chk['z_score']:.1f}). Try a plain background with one item.")
            except Exception:
                pass
            bar.progress((i + 1) / len(files))
        st.success("Done! Check the tags below and fix any mistakes.")

    garments = db.list_garments()
    slot_filter = st.segmented_control("Show", ["all", "top", "bottom", "onepiece", "layer", "shoes"],
                                       default="all") if hasattr(st, "segmented_control") else "all"
    shown = [g for g in garments if slot_filter in (None, "all") or g["slot"] == slot_filter]
    cols = st.columns(5)
    for i, g in enumerate(shown):
        with cols[i % 5]:
            st.image(g["image_path"], width=170)
            st.caption(f"#{g['id']} · {describe(g)} · formality {g['formality']}")
            with st.expander("Edit"):
                cats = list(CATEGORIES)
                cat = st.selectbox("Category", cats, index=cats.index(g["category"]), key=f"c{g['id']}")
                pat = st.selectbox("Pattern", PATTERNS, index=PATTERNS.index(g["pattern"]), key=f"p{g['id']}")
                cols_sel = st.multiselect("Colours", list(PALETTE), default=g["colors"], key=f"k{g['id']}")
                form = st.slider("Formality", 1, 5, g["formality"], key=f"f{g['id']}")
                a, b = st.columns(2)
                if a.button("Save", key=f"s{g['id']}"):
                    slot, _, warmth = CATEGORIES[cat]
                    name = " ".join([(cols_sel or ["?"])[0]] + ([pat] if pat != "solid" else []) + [cat])
                    db.update_garment(g["id"], category=cat, slot=slot, warmth=warmth, pattern=pat,
                                      colors=cols_sel or g["colors"], formality=form, name=name)
                    st.rerun()
                if b.button("Delete", key=f"d{g['id']}"):
                    db.delete_garment(g["id"])
                    st.rerun()
    if not garments:
        st.info("Your wardrobe is empty. Upload a few tops, bottoms and shoes to get started.")


# ---------------------------------------------------------------- outfit card
def outfit_card(o: dict, key: str, day: date):
    with st.container(border=True):
        st.markdown(f"**{o['title']}**  ·  score {o['score']:.2f}")
        pics = st.columns(len(o["items"]))
        for c, g in zip(pics, o["items"]):
            c.image(g["image_path"], width=130)
            c.caption(describe(g))
        st.write(o["reason"])
        with st.expander("Why this score?"):
            st.json(o["parts"])
        b1, b2, b3 = st.columns(3)
        if b1.button("👍 Like", key=f"l{key}"):
            stylist.feedback(o, "like", day)
            st.toast("Thanks! Your style profile was updated.")
        if b2.button("👎 Skip", key=f"x{key}"):
            stylist.feedback(o, "skip", day)
            st.toast("Got it, fewer outfits like this.")
        if b3.button("✅ Wear it", key=f"w{key}"):
            stylist.feedback(o, "worn", day)
            st.toast("Marked as worn. It won't repeat for a few days.")


# ---------------------------------------------------------------- stylist
with tab_s:
    st.subheader("Ask your stylist")
    c1, c2 = st.columns([3, 1])
    prompt = c1.text_input("What's the occasion?", placeholder="e.g. Placement interview tomorrow, want to look sharp")
    day = c2.date_input("When?", date.today())
    if st.button("Style me ✨", type="primary") and prompt:
        try:
            with st.spinner("Picking outfits from your wardrobe..."):
                ss.recs = stylist.recommend(prompt, city, day, clip_model())
                ss.recs["day"] = day
        except ValueError as e:
            st.error(str(e))
    r = ss.recs
    if r:
        wx, it = r["weather"], r["intent"]
        st.caption(f"🌤 {wx['temp_c']} °C, {wx['description']}, rain {wx['rain_prob']:.0%} · "
                   f"🎯 {it['occasion']} (formality {it['formality']}/5, parsed by {it['source']}) · "
                   f"⏱ {sum(r['timings'].values()):.2f} s")
        if not r["outfits"]:
            st.warning("No complete outfit matched. Add more tops/bottoms with a similar formality level.")
        for i, o in enumerate(r["outfits"]):
            outfit_card(o, f"r{i}", r["day"])

# ---------------------------------------------------------------- week planner
with tab_p:
    st.subheader("Plan the week")
    c1, c2 = st.columns([3, 1])
    wprompt = c1.text_input("Typical day", "college classes")
    start = c2.date_input("Starting", date.today() + timedelta(days=1), key="plan_start")
    if st.button("Plan my week"):
        try:
            with st.spinner("Planning 7 outfits without repeats..."):
                plan = stylist.plan_week(wprompt, city, start, clip_model())
            cols = st.columns(7)
            for c, p in zip(cols, plan):
                c.markdown(f"**{p['day']:%a %d}**  \n{p['weather']['temp_c']} °C")
                if p["outfit"]:
                    for g in p["outfit"]["items"]:
                        c.image(g["image_path"], width=90)
                else:
                    c.caption("No outfit")
        except ValueError as e:
            st.error(str(e))

# ---------------------------------------------------------------- design studio (VAE)
with tab_d:
    st.subheader("Design Studio: Variational Autoencoder")
    st.caption("A VAE trained on Fashion-MNIST (70,000 clothing images). It compresses each garment into "
               "16 numbers, and that one latent space powers every feature below.")
    if not genstudio.trained():
        st.info("The VAE isn't trained yet. Run `python -m genmodels.train` (about 10-20 minutes on a laptop), "
                "then refresh this page. See README section 7.")
    else:
        met = genstudio.metrics()
        if met:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Test -ELBO", f"{met.get('neg_elbo', 0):.1f} nats")
            a2.metric("Reconstruction MSE", f"{met.get('recon_mse', 0):.4f}")
            a3.metric("KL term", f"{met.get('kl', 0):.1f} nats")
            a4.metric("FID (new designs)", f"{met['fid']:.1f}" if "fid" in met else "run evaluate")

        garments = db.list_garments()
        pick = lambda lbl, key, idx=0: garments[st.selectbox(
            lbl, range(len(garments)), index=min(idx, len(garments) - 1),
            format_func=lambda i: f"#{garments[i]['id']} {describe(garments[i])}", key=key)]

        st.markdown("#### 1 · Generate new garment designs")
        c1, c2 = st.columns([1, 1])
        n = c1.slider("How many", 8, 32, 16, step=8)
        temp = c2.slider("Creativity (sampling temperature)", 0.5, 1.5, 1.0, 0.1,
                         help="Higher = more unusual designs, lower = safer, more typical ones")
        if st.button("Generate designs"):
            st.image(genstudio.to_pil(genstudio.generate(n, temp), nrow=8),
                     caption="Decoded from random points z ~ N(0, I) in the VAE latent space")

        st.markdown("#### 2 · Design mixer (latent interpolation)")
        if len(garments) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                ga = pick("Garment A", "mix_a")
            with c2:
                gb = pick("Garment B", "mix_b", 1)
            if st.button("Mix them"):
                st.image(genstudio.to_pil(genstudio.interpolate(ga["image_path"], gb["image_path"], 10), nrow=10),
                         caption="A → B: decoding points on the straight line between their latent codes")
        else:
            st.caption("Add at least 2 garments to your wardrobe.")

        if garments:
            st.markdown("#### 3 · Photo clean-up (reconstruction)")
            gd = pick("Garment", "dn")
            noise = st.slider("Noise to add", 0.0, 0.6, 0.3, 0.05)
            if st.button("Add noise and rebuild"):
                st.image(genstudio.to_pil(genstudio.clean_up(gd["image_path"], noise), nrow=3, scale=6),
                         caption="original · noisy · rebuilt by the VAE from its latent code")

            st.markdown("#### 4 · Find similar clothes (latent-space search)")
            if len(garments) >= 2:
                gs = pick("Find clothes similar to", "sim")
                cols = st.columns(4)
                for col, (g, sim) in zip(cols, genstudio.similar(gs, garments, k=4)):
                    col.image(g["image_path"], width=100, caption=f"{describe(g)} · {sim:.2f}")

            st.markdown("#### 5 · Upload quality check (reconstruction error)")
            gq = pick("Check garment photo", "qc")
            res = genstudio.photo_check(gq["image_path"])
            c1, c2 = st.columns([1, 3])
            c1.image(genstudio.to_pil(genstudio.to_fmnist(gq["image_path"]), nrow=1, scale=4),
                     caption="as the VAE sees it")
            if res:
                c2.write(f"Reconstruction error {res['recon_error']:.4f} · z-score {res['z_score']:.1f}")
                (c2.warning if res["unusual"] else c2.success)(
                    "Unusual photo — may be tagged wrongly." if res["unusual"] else "Looks like a normal garment photo.")
                c2.caption("z-score compares the error with real Fashion-MNIST test clothes; above 4 is flagged.")
        else:
            st.caption("Add garments on the Wardrobe tab to use features 3-5 on your own clothes.")


# ---------------------------------------------------------------- insights
with tab_i:
    st.subheader("Wardrobe insights")
    garments = db.list_garments()
    counts = db.wear_counts()
    fb = db.feedback_stats()
    a, b, c, d = st.columns(4)
    a.metric("Garments", len(garments))
    b.metric("Outfits worn", fb.get("worn", 0))
    total = sum(fb.get(x, 0) for x in ("like", "skip", "worn"))
    c.metric("Acceptance rate", f"{(fb.get('like', 0) + fb.get('worn', 0)) / total:.0%}" if total else "–")
    d.metric("Never worn", sum(1 for g in garments if g["id"] not in counts))
    if garments:
        st.markdown("**Items by type**")
        st.bar_chart({s: sum(1 for g in garments if g["slot"] == s) for s in ("top", "bottom", "onepiece", "layer", "shoes")})
        never = [g for g in garments if g["id"] not in counts][:10]
        if never:
            st.markdown("**Forgotten clothes — try wearing these**")
            cols = st.columns(len(never))
            for c, g in zip(cols, never):
                c.image(g["image_path"], width=100)