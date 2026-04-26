from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Global Food Price Dashboard 2026",
    page_icon="🌍",
    layout="wide",
)

try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path("/content")

DEFAULT_DATA_PATH = BASE_DIR / "hdx_hapi_food_price_global_2026.csv"


@st.cache_data(show_spinner=False)
def load_data(data_source) -> pd.DataFrame:
    df = pd.read_csv(data_source, low_memory=False)
    df["period_start"] = pd.to_datetime(df["reference_period_start"], dayfirst=True, errors="coerce")
    df["period_end"] = pd.to_datetime(df["reference_period_end"], dayfirst=True, errors="coerce")
    df["period_label"] = df["period_start"].dt.strftime("%Y-%m")
    df["usd_price"] = pd.to_numeric(df["usd_price"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    fill_unknown = [
        "price_type",
        "commodity_category",
        "commodity_name",
        "unit",
        "location_code",
        "market_name",
    ]
    for col in fill_unknown:
        df[col] = df[col].fillna("Unknown")
    return df


def apply_filters(
    df: pd.DataFrame,
    selected_periods: list[str],
    selected_countries: list[str],
    price_type: str,
    hrp_filter: str,
    gho_filter: str,
    category_filter: str,
) -> pd.DataFrame:
    filtered = df.copy()

    if selected_periods:
        filtered = filtered[filtered["period_label"].isin(selected_periods)]
    if selected_countries:
        filtered = filtered[filtered["location_code"].isin(selected_countries)]
    if price_type != "All":
        filtered = filtered[filtered["price_type"] == price_type]
    if hrp_filter != "All":
        filtered = filtered[filtered["has_hrp"] == hrp_filter]
    if gho_filter != "All":
        filtered = filtered[filtered["in_gho"] == gho_filter]
    if category_filter != "All":
        filtered = filtered[filtered["commodity_category"] == category_filter]

    return filtered


def format_number(value: float | int) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:,.2f}"


def show_overview(filtered_df: pd.DataFrame) -> None:
    st.header("1. Dataset overview")
    st.write(
        "This section provides a high-level overview of the filtered data so users can check "
        "coverage before moving into like-for-like price comparisons."
    )

    if filtered_df.empty:
        st.warning("No rows match the current filters. Please relax the filter selection.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", format_number(len(filtered_df)))
    c2.metric("Countries", format_number(filtered_df["location_code"].nunique()))
    c3.metric("Markets", format_number(filtered_df["market_code"].nunique()))
    c4.metric("Median USD price", format_number(filtered_df["usd_price"].median()))

    with st.container():
        st.subheader("Monthly coverage")
        monthly_counts = (
            filtered_df.groupby("period_label")
            .size()
            .rename("Rows")
            .sort_index()
            .to_frame()
        )
        st.line_chart(monthly_counts)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Commodity categories")
        category_counts = (
            filtered_df["commodity_category"]
            .value_counts()
            .rename_axis("Commodity category")
            .reset_index(name="Rows")
            .set_index("Commodity category")
        )
        st.bar_chart(category_counts)

    with col2:
        st.subheader("Price type distribution")
        type_counts = (
            filtered_df["price_type"]
            .value_counts()
            .rename_axis("Price type")
            .reset_index(name="Rows")
            .set_index("Price type")
        )
        st.bar_chart(type_counts)

    st.subheader("HRP coverage over time")
    hrp_by_month = (
        filtered_df.groupby(["period_label", "has_hrp"])
        .size()
        .rename("Rows")
        .reset_index()
        .pivot(index="period_label", columns="has_hrp", values="Rows")
        .fillna(0)
        .sort_index()
    )
    if not hrp_by_month.empty:
        st.area_chart(hrp_by_month)


def show_comparable_analysis(filtered_df: pd.DataFrame, min_country_obs: int) -> None:
    st.header("2. Comparable price explorer")
    st.write(
        "This section narrows the analysis to one commodity and one unit so that the charts compare "
        "like with like instead of mixing incompatible units."
    )

    if filtered_df.empty:
        st.warning("No rows are available for comparison under the current filters.")
        return

    available_commodities = sorted(filtered_df["commodity_name"].dropna().unique().tolist())
    default_commodity = "Sugar" if "Sugar" in available_commodities else available_commodities[0]
    commodity_choice = st.selectbox(
        "Choose one commodity",
        available_commodities,
        index=available_commodities.index(default_commodity),
    )

    commodity_df = filtered_df[filtered_df["commodity_name"] == commodity_choice].copy()
    available_units = sorted(commodity_df["unit"].dropna().unique().tolist())
    default_unit = "KG" if "KG" in available_units else available_units[0]
    unit_choice = st.selectbox(
        "Choose one unit",
        available_units,
        index=available_units.index(default_unit),
    )

    comparable_df = commodity_df[commodity_df["unit"] == unit_choice].copy()

    if comparable_df.empty:
        st.warning("No comparable rows were found for the selected commodity and unit.")
        return

    obs_by_country = comparable_df.groupby("location_code").size()
    eligible_countries = obs_by_country[obs_by_country >= min_country_obs].index.tolist()
    comparable_df = comparable_df[comparable_df["location_code"].isin(eligible_countries)].copy()

    if comparable_df.empty:
        st.warning(
            "No countries meet the current minimum observation threshold. Reduce the slider in the sidebar."
        )
        return

    st.success(
        f"Comparable analysis is active for {commodity_choice} measured in {unit_choice}. "
        f"This view contains {len(comparable_df):,} rows across {comparable_df['location_code'].nunique()} countries."
    )

    trend_df = (
        comparable_df.groupby(["period_label", "location_code"])["usd_price"]
        .mean()
        .reset_index()
    )
    top_countries = comparable_df["location_code"].value_counts().head(8).index.tolist()
    trend_pivot = (
        trend_df[trend_df["location_code"].isin(top_countries)]
        .pivot(index="period_label", columns="location_code", values="usd_price")
        .sort_index()
    )

    st.subheader("Average USD price by month")
    if not trend_pivot.empty:
        st.line_chart(trend_pivot)
    else:
        st.info("Not enough data to draw the monthly trend chart.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average USD price by country")
        country_rank = (
            comparable_df.groupby("location_code")["usd_price"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .to_frame(name="Average USD price")
        )
        st.bar_chart(country_rank)

    with col2:
        st.subheader("Price spread over time")
        spread = comparable_df.groupby("period_label")["usd_price"].agg(["min", "median", "max"]).sort_index()
        st.area_chart(spread)

    st.subheader("Filtered comparable data preview")
    st.dataframe(
        comparable_df[
            [
                "location_code",
                "market_name",
                "commodity_name",
                "unit",
                "price_type",
                "usd_price",
                "period_label",
            ]
        ].head(50),
        use_container_width=True,
    )


def show_market_map(filtered_df: pd.DataFrame) -> None:
    st.header("3. Markets and map")
    st.write(
        "The tutorial shows that Streamlit can display maps using latitude and longitude, "
        "so this section uses market coordinates to show spatial coverage."
    )

    if filtered_df.empty:
        st.warning("No rows are available for the map under the current filters.")
        return

    map_df = filtered_df.dropna(subset=["lat", "lon"]).copy()

    if map_df.empty:
        st.error("The current filtered data has no valid latitude and longitude values.")
        return

    st.subheader("Market locations")
    st.map(map_df[["lat", "lon"]].drop_duplicates())

    st.subheader("Top markets by average USD price")
    market_rank = (
        map_df.groupby("market_name")["usd_price"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .to_frame(name="Average USD price")
    )
    st.bar_chart(market_rank)

    st.subheader("Rows by country")
    country_rows = map_df["location_code"].value_counts().head(15).to_frame(name="Rows")
    st.bar_chart(country_rows)


def show_data_quality(filtered_df: pd.DataFrame) -> None:
    st.header("4. Data quality and download")
    st.write(
        "This section highlights missing values and provides the filtered data as a downloadable CSV file."
    )

    missing_usd = int(filtered_df["usd_price"].isna().sum()) if "usd_price" in filtered_df else 0
    missing_price = int(filtered_df["price"].isna().sum()) if "price" in filtered_df else 0
    missing_coords = int(filtered_df[["lat", "lon"]].isna().any(axis=1).sum()) if not filtered_df.empty else 0

    q1, q2, q3 = st.columns(3)
    q1.metric("Missing USD prices", format_number(missing_usd))
    q2.metric("Missing local prices", format_number(missing_price))
    q3.metric("Rows missing coordinates", format_number(missing_coords))

    if filtered_df.empty:
        st.warning("There is no filtered data to preview or download.")
        return

    st.info(
        "Interpret prices carefully. Different commodities and units should not be compared directly "
        "unless the app has been narrowed to a single commodity and unit."
    )

    st.subheader("Filtered data table")
    st.dataframe(filtered_df.head(100), use_container_width=True)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_bytes,
        file_name="filtered_food_prices.csv",
        mime="text/csv",
    )

    if st.button("Celebrate successful testing"):
        st.balloons()


def main() -> None:
    st.title("Global Food Price Dashboard 2026")
    st.caption("Google Colab friendly Streamlit coursework app using the HDX food price dataset.")

    st.sidebar.header("Data source")
    uploaded_file = st.sidebar.file_uploader("Upload the CSV if it is not in the app folder", type=["csv"])

    with st.spinner("Loading dataset and preparing the dashboard..."):
        if uploaded_file is not None:
            df = load_data(uploaded_file)
        elif DEFAULT_DATA_PATH.exists():
            df = load_data(DEFAULT_DATA_PATH)
        else:
            st.error(
                "Dataset not found. Upload hdx_hapi_food_price_global_2026.csv in the sidebar or place it in the same folder as streamlit_app.py."
            )
            st.stop()

    st.success(
        f"Dataset loaded successfully: {len(df):,} rows, {df.shape[1]} columns and "
        f"{df['location_code'].nunique()} countries."
    )

    st.sidebar.header("Dashboard controls")
    section = st.sidebar.radio(
        "Choose a dashboard section",
        [
            "Overview",
            "Comparable price explorer",
            "Markets and map",
            "Data quality and download",
        ],
    )

    period_options = sorted(df["period_label"].dropna().unique().tolist())
    selected_periods = st.sidebar.multiselect("Reference period", period_options, default=period_options)

    country_options = sorted(df["location_code"].dropna().unique().tolist())
    selected_countries = st.sidebar.multiselect("Country code", country_options, default=[])

    price_type = st.sidebar.selectbox("Price type", ["All"] + sorted(df["price_type"].dropna().unique().tolist()))
    hrp_filter = st.sidebar.selectbox("Humanitarian Response Plan", ["All"] + sorted(df["has_hrp"].dropna().unique().tolist()))
    gho_filter = st.sidebar.selectbox("Global Humanitarian Overview", ["All"] + sorted(df["in_gho"].dropna().unique().tolist()))
    category_filter = st.sidebar.selectbox("Commodity category", ["All"] + sorted(df["commodity_category"].dropna().unique().tolist()))
    min_country_obs = st.sidebar.slider("Minimum observations per country", 1, 25, 5)

    filtered_df = apply_filters(
        df,
        selected_periods,
        selected_countries,
        price_type,
        hrp_filter,
        gho_filter,
        category_filter,
    )

    if st.sidebar.checkbox("Show raw filtered data preview"):
        st.sidebar.write(filtered_df.head(10))

    st.markdown(
        """
        **How to use this app**
        1. Apply the sidebar filters.
        2. Start with the overview.
        3. Move to the comparable price explorer for like-for-like analysis.
        """
    )

    if section == "Overview":
        show_overview(filtered_df)
    elif section == "Comparable price explorer":
        show_comparable_analysis(filtered_df, min_country_obs)
    elif section == "Markets and map":
        show_market_map(filtered_df)
    else:
        show_data_quality(filtered_df)


if __name__ == "__main__":
    main()
