#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
#   ÁLTALÁNOS DIZÁJN BEÁLLÍTÁSOK
# -----------------------------
plt.style.use("dark_background")

BG_DASH = "#020617"      # fő háttér (dashboard – ezt a theme is adja)
BG_PANEL = "#111827"     # grafikonpanelek háttere
TEXT_COLOR = "#e5e7eb"   # világos szöveg

COLOR_FULL = "#4b5563"       # teljes sokaság oszlop
COLOR_FULL_EDGE = "#9ca3af"

COLOR_FILTER = "#f97316"     # szűrt eloszlás / korcsoportos oszlop
COLOR_FILTER_EDGE = "#fed7aa"

COLOR_EDU = "#22c55e"        # átlagéletkor végzettség szerint
BOX_COLORS = ["#38bdf8", "#fb923c"]  # Férfi / Nő boxplot + nemek aránya

# -----------------------------
#   OLDAL CÍM + RÖVID LEÍRÁS
# -----------------------------
st.title("Demográfiai dashboard")
st.subheader("Kor eloszlása")

st.markdown(
    """
Ez a mini dashboard a **kor eloszlását** mutatja.
Bal oldalt szűrheted a mintát végzettség, nem és kor tartomány szerint,
a grafikonok és a mutatók pedig azonnal ehhez igazodnak.
"""
)

# -----------------------------
#   ADAT BEOLVASÁS
# -----------------------------
df = pd.read_excel("cx_filter.xlsx")

# Csak olyan sorok, ahol van kor
df = df.dropna(subset=["kor"])

# -----------------------------
#   OLDALSÁV SZŰRŐK
# -----------------------------
st.sidebar.header("Szűrők")

# Végzettség
vegzok = df["vegz"].dropna().unique()
selected_vegz = st.sidebar.multiselect("Végzettség", vegzok, default=vegzok)

# Nem
nemek = df["nem"].dropna().unique()
selected_nem = st.sidebar.multiselect("Nem", nemek, default=nemek)

# Kor tartomány (slider)
min_age = int(df["kor"].min())
max_age = int(df["kor"].max())
selected_age_range = st.sidebar.slider(
    "Kor tartomány",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age),
    step=1,
)

# Hisztogram oszlopainak száma
bins = st.sidebar.number_input(
    "Hisztogram oszlopainak száma",
    min_value=1,
    max_value=60,
    value=20,
    step=1,
)

# -----------------------------
#   ADATOK SZŰRÉSE
# -----------------------------
all_values = df["kor"].dropna()

filtered_df = df[
    (df["vegz"].isin(selected_vegz))
    & (df["nem"].isin(selected_nem))
    & (df["kor"].between(selected_age_range[0], selected_age_range[1]))
]

filtered_values = filtered_df["kor"].dropna()

# Aktív szűrők rövid szövege
filters_text = (
    f"Végzettség: {', '.join(selected_vegz) if len(selected_vegz) > 0 else 'nincs'} | "
    f"Nem: {', '.join(selected_nem) if len(selected_nem) > 0 else 'nincs'} | "
    f"Kor: {selected_age_range[0]}–{selected_age_range[1]} év"
)
st.caption(f"**Aktív szűrők:** {filters_text}")

st.subheader("Kor eloszlás")

if filtered_df.empty:
    st.warning("Nincs megjeleníthető adat a kiválasztott szűrők alapján.")
else:
    # -----------------------------
    #   FELSŐ "MUTATÓK" (KÁRTYÁK)
    # -----------------------------
    avg_all = all_values.mean()
    avg_filtered = filtered_values.mean()
    median_filtered = filtered_values.median()
    min_filtered = filtered_values.min()
    max_filtered = filtered_values.max()
    diff_avg = avg_filtered - avg_all

    # delta szöveg – ne legyen +0.0
    if abs(diff_avg) < 0.05:
        delta_display = "≈ 0 év"
    else:
        delta_display = f"{diff_avg:+.1f} év"

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Elemszám", f"{len(filtered_values)} fő")
    col2.metric("Átlagéletkor", f"{avg_filtered:.1f} év", delta_display)
    col3.metric("Medián", f"{median_filtered:.1f} év")
    col4.metric("Kor tartomány", f"{min_filtered:.0f} – {max_filtered:.0f} év")

    # -----------------------------
    #   FŐ HISZTOGRAM (TELJES VS. SZŰRT)
    # -----------------------------
    min_val = min(all_values.min(), filtered_values.min())
    max_val = max(all_values.max(), filtered_values.max())

    edges = np.linspace(min_val, max_val, int(bins) + 1)

    counts_all, _ = np.histogram(all_values, bins=edges)
    counts_filtered, _ = np.histogram(filtered_values, bins=edges)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_DASH)
    ax.set_facecolor(BG_PANEL)

    bin_widths = np.diff(edges)
    centers = edges[:-1] + bin_widths / 2

    # Háttér – teljes sokaság
    ax.bar(
        centers,
        counts_all,
        width=bin_widths * 0.9,
        color=COLOR_FULL,
        alpha=0.5,
        edgecolor=COLOR_FULL_EDGE,
        label="Teljes sokaság",
    )

    # Előtér – szűrt minta
    ax.bar(
        centers,
        counts_filtered,
        width=bin_widths * 0.7,
        color=COLOR_FILTER,
        alpha=0.85,
        edgecolor=COLOR_FILTER_EDGE,
        label="Szűrt eloszlás",
    )

    ax.set_xlabel("Kor", color=TEXT_COLOR)
    ax.set_ylabel("Gyakoriság", color=TEXT_COLOR)
    ax.set_title("Kor szerinti eloszlás (szűrt vs. teljes sokaság)", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.legend(facecolor=BG_PANEL, edgecolor=COLOR_FULL_EDGE)

    st.pyplot(fig)

    # -----------------------------
    #   RÖVID SZÖVEGES ÖSSZEFOGLALÓ
    # -----------------------------
    st.subheader("Rövid szöveges összefoglaló")

    if abs(diff_avg) < 0.05:
        summary_text = (
            f"A kiválasztott szűrők alapján **{len(filtered_values)} fő** szerepel a mintában.  \n"
            f"A szűrt csoport átlagéletkora **{avg_filtered:.1f} év**, "
            f"ami gyakorlatilag **megegyezik** a teljes sokaság átlagával (**{avg_all:.1f} év**)."
        )
    elif diff_avg > 0:
        summary_text = (
            f"A kiválasztott szűrők alapján **{len(filtered_values)} fő** szerepel a mintában.  \n"
            f"A szűrt csoport átlagéletkora **{avg_filtered:.1f} év**, "
            f"ami **{diff_avg:.1f} évvel magasabb** a teljes sokaság átlagánál (**{avg_all:.1f} év**)."
        )
    else:
        summary_text = (
            f"A kiválasztott szűrők alapján **{len(filtered_values)} fő** szerepel a mintában.  \n"
            f"A szűrt csoport átlagéletkora **{avg_filtered:.1f} év**, "
            f"ami **{abs(diff_avg):.1f} évvel alacsonyabb** a teljes sokaság átlagánál (**{avg_all:.1f} év**)."
        )

    st.markdown(summary_text)

    # -----------------------------
    #   KORCSOPORTOS MEGOSZLÁS
    # -----------------------------
    st.subheader("Korcsoportok megoszlása")

    # Korcsoport-határok – igény szerint módosítható
    age_bins = [18, 26, 36, 46, 56, max_age + 1]
    age_labels = ["18–25", "26–35", "36–45", "46–55", "56+"]

    age_groups = pd.cut(filtered_values, bins=age_bins, labels=age_labels, right=False)
    age_counts = age_groups.value_counts().reindex(age_labels, fill_value=0)
    age_percent = (age_counts / len(filtered_values) * 100).round(1)

    fig_age, ax_age = plt.subplots(figsize=(8, 4))
    fig_age.patch.set_facecolor(BG_DASH)
    ax_age.set_facecolor(BG_PANEL)

    ax_age.bar(
        age_labels,
        age_percent.values,
        color=COLOR_FILTER,
        edgecolor=COLOR_FILTER_EDGE,
        alpha=0.9,
    )

    ax_age.set_xlabel("Korcsoport", color=TEXT_COLOR)
    ax_age.set_ylabel("Arány (%)", color=TEXT_COLOR)
    ax_age.set_title("Korcsoportos megoszlás a szűrt mintában", color=TEXT_COLOR)
    ax_age.tick_params(colors=TEXT_COLOR)

    for i, v in enumerate(age_percent.values):
        ax_age.text(i, v + 0.5, f"{v:.1f}%", ha="center", color=TEXT_COLOR, fontsize=9)

    st.pyplot(fig_age)

    # -----------------------------
    #   ÁTLAGÉLETKOR VÉGZETTSÉG SZERINT + NEMEK ARÁNYA
    # -----------------------------
    st.subheader("Végzettség és nemek szerinti áttekintés")

    col_left, col_right = st.columns(2)

    # ---- Bal: Átlagéletkor végzettség szerint ----
    with col_left:
        st.markdown("**Átlagéletkor végzettség szerint**")

        edu_stats = (
            filtered_df.groupby("vegz")["kor"]
            .mean()
            .round(1)
            .sort_values()
        )

        fig_edu, ax_edu = plt.subplots(figsize=(5, 4))
        fig_edu.patch.set_facecolor(BG_DASH)
        ax_edu.set_facecolor(BG_PANEL)

        ax_edu.bar(
            edu_stats.index.astype(str),
            edu_stats.values,
            color=COLOR_EDU,
            edgecolor="#bbf7d0",
            alpha=0.9,
        )

        ax_edu.set_xlabel("Végzettség", color=TEXT_COLOR)
        ax_edu.set_ylabel("Átlagéletkor", color=TEXT_COLOR)
        ax_edu.tick_params(colors=TEXT_COLOR)

        for i, v in enumerate(edu_stats.values):
            ax_edu.text(i, v + 0.3, f"{v:.1f}", ha="center", color=TEXT_COLOR, fontsize=9)

        st.pyplot(fig_edu)

    # ---- Jobb: Nemek aránya ----
    with col_right:
        st.markdown("**Nemek aránya a szűrt mintában**")

        gender_counts = (
            filtered_df["nem"].value_counts(normalize=True) * 100
        ).round(1)

        # biztosítsuk, hogy rendezetten jöjjön: pl. Férfi, Nő vagy adat szerinti
        genders = list(gender_counts.index.astype(str))
        values = gender_counts.values

        fig_gender, ax_gender = plt.subplots(figsize=(5, 4))
        fig_gender.patch.set_facecolor(BG_DASH)
        ax_gender.set_facecolor(BG_PANEL)

        colors = BOX_COLORS[: len(genders)]

        ax_gender.bar(
            genders,
            values,
            color=colors,
            edgecolor=TEXT_COLOR,
            alpha=0.9,
        )

        ax_gender.set_xlabel("Nem", color=TEXT_COLOR)
        ax_gender.set_ylabel("Arány (%)", color=TEXT_COLOR)
        ax_gender.tick_params(colors=TEXT_COLOR)

        for i, v in enumerate(values):
            ax_gender.text(i, v + 0.5, f"{v:.1f}%", ha="center", color=TEXT_COLOR, fontsize=9)

        st.pyplot(fig_gender)

    # -----------------------------
    #   RÉSZLETES STATISZTIKA (EXPANDERBEN)
    # -----------------------------
    with st.expander("Részletes statisztika végzettség és nem szerint"):
        group_stats = (
            filtered_df.groupby(["vegz", "nem"])["kor"]
            .agg(elemszám="count", átlag="mean", szórás="std")
            .round({"átlag": 1, "szórás": 1})
        )
        st.dataframe(
            group_stats.reset_index(),
            use_container_width=True,
        )

    # -----------------------------
    #   ÉLETKOR NEMEK SZERINT (BOXPLOT)
    # -----------------------------
    st.subheader("Életkor eloszlása nemek szerint")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    fig2.patch.set_facecolor(BG_DASH)
    ax2.set_facecolor(BG_PANEL)

    # patch_artist=True → kitöltött dobozok
    filtered_df.boxplot(
        column="kor",
        by="nem",
        ax=ax2,
        patch_artist=True,
    )

    # Dobozok kitöltése két külön színnel (Férfi / Nő)
    for patch, color in zip(ax2.artists, BOX_COLORS):
        patch.set_facecolor(color)
        patch.set_edgecolor(TEXT_COLOR)
        patch.set_alpha(0.85)

    # Vonalak (median, whiskerek, stb.) színezése
    for line in ax2.lines:
        line.set_color(TEXT_COLOR)

    ax2.set_title(
        "Életkor eloszlása nemek szerint (szűrt minta)",
        color=TEXT_COLOR,
    )
    ax2.set_xlabel("Nem", color=TEXT_COLOR)
    ax2.set_ylabel("Kor", color=TEXT_COLOR)
    ax2.tick_params(colors=TEXT_COLOR)
    plt.suptitle("")

    st.pyplot(fig2)

    st.caption(
        "A doboz a középső 50%-ot mutatja, a vízszintes vonal a medián életkort jelzi."
    )

    # -----------------------------
    #   SZŰRT ADATOK EXPORTJA (EXPANDERBEN)
    # -----------------------------
    with st.expander("Szűrt adatok exportja"):
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Szűrt adatok letöltése CSV-ként",
            data=csv_data,
            file_name="szurt_adatok.csv",
            mime="text/csv",
        )
