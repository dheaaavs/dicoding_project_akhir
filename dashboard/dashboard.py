import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

# Set style to dark with navy palette
sns.set(style='dark', palette='muted')

# Define color palette
NAVY_BLUE = "#001F3F"
LIGHT_BLUE = "#0074D9"
DARK_BLUE = "#001F3F"

# Helper functions
def create_daily_orders_df(df):
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        "order_id": "nunique",
        "delivery_time": "mean" 
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        "order_id": "order_count", 
        "delivery_time": "avg_delivery_time"  
    }, inplace=True)
    
    return daily_orders_df

def group_and_count_orders_by_status(df):
    order_group_df = df.groupby(by="order_status").order_id.nunique().reset_index()
    order_group_df.rename(columns={"order_id": "order_count"}, inplace=True)
    order_group_df = order_group_df.sort_values(by="order_count", ascending=False)
    return order_group_df

def group_by_city_and_status(df):
    city_status_group = df.groupby(by=["customer_city", "order_status"]).agg({
        "order_id": "count"
    }).reset_index()
    city_status_group.rename(columns={"order_id": "order_count"}, inplace=True)
    return city_status_group

def create_rfm_df(df):
    required_columns = ['customer_id', 'order_purchase_timestamp', 'order_id']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' does not exist in the dataset.")
    
    rfm_df = df.groupby(by="customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max",  # Recency calculation
        "order_id": "nunique"               # Frequency calculation
    })
    
    rfm_df["monetary"] = df.groupby("customer_id")["order_id"].count().values
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    
    # Calculate recency
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
    
    return rfm_df

# Load cleaned data
all_df = pd.read_csv("dashboard/all_data.csv")
datetime_columns = ["order_purchase_timestamp", "order_delivered_customer_date"]
all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)

for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

# Filter data
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

# Extracting unique years from the dataset
years = all_df["order_purchase_timestamp"].dt.year.unique()

with st.sidebar:
    st.title('Customers Dashboard') 
    st.image("https://raw.githubusercontent.com/dheaaavs/dataset_project_dicoding/main/baju.png")
    
    selected_year = st.selectbox(
        label='Select Year', 
        options=sorted(years),
        index=len(years) - 1  # Default to the most recent year
    )
    
    # Date input for selecting the range
    start_date, end_date = st.date_input(
        label='Rentang Waktu', 
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

# Filtering main DataFrame based on selected year and date range
main_df = all_df[
    (all_df["order_purchase_timestamp"].dt.year == selected_year) &
    (all_df["order_purchase_timestamp"] >= str(start_date)) & 
    (all_df["order_purchase_timestamp"] <= str(end_date))
]

# Preparing dataframes
daily_orders_df = create_daily_orders_df(main_df)
bycity_status_df = group_by_city_and_status(main_df)
order_group_df = group_and_count_orders_by_status(main_df)
rfm_df = create_rfm_df(main_df)

# Dashboard header
st.header('Customers Collection Dashboard :sparkles::heart::moon:')
st.subheader('Daily Orders')

col1, col2 = st.columns(2)

with col1:
    total_orders = daily_orders_df.order_count.sum()
    st.metric("Total Orders", value=total_orders, delta=0)

with col2:
    avg_delivery_time = daily_orders_df.avg_delivery_time.mean()
    st.metric("Avg Delivery Time", value=f"{avg_delivery_time:.2f} days", delta=0)

# Plot number of daily orders
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_orders_df["order_purchase_timestamp"],
    daily_orders_df["order_count"],
    marker='o', 
    linewidth=2,
    color=LIGHT_BLUE
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
ax.set_title("Daily Orders Over Time", fontsize=24, color=NAVY_BLUE)
ax.set_xlabel("Date", fontsize=18, color=NAVY_BLUE)
ax.set_ylabel("Number of Orders", fontsize=18, color=NAVY_BLUE)

st.pyplot(fig)

# Visualisasi Orders by City and Status
delivered_df = bycity_status_df[bycity_status_df["order_status"] == "delivered"]

# Get top 10 cities by order count
top_10_cities = delivered_df.groupby("customer_city")["order_count"].sum().nlargest(10).index

# Filter dataframe to show only top 10 cities
top_10_delivered_df = delivered_df[delivered_df["customer_city"].isin(top_10_cities)]

# Plot the top 10 cities with delivered orders
st.subheader('Top 10 Cities by Delivered Orders')
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="order_count", y="customer_city", hue="order_status", data=top_10_delivered_df, palette='Blues_r', ax=ax)
ax.set_title("Top 10 Cities by Delivered Orders", fontsize=20, color=NAVY_BLUE)
ax.set_xlabel("Number of Orders", fontsize=15, color=NAVY_BLUE)
ax.set_ylabel("City", fontsize=15, color=NAVY_BLUE)
ax.legend(title="Order Status", fontsize=12)
st.pyplot(fig)

# Visualisasi Order Group by Status
st.subheader('Order Group by Status')
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x="order_count", y="order_status", data=order_group_df, palette='Blues_r', ax=ax)
ax.set_title("Order Count by Status", fontsize=20, color=NAVY_BLUE)
ax.set_xlabel("Number of Orders", fontsize=15, color=NAVY_BLUE)
ax.set_ylabel("Order Status", fontsize=15, color=NAVY_BLUE)
st.pyplot(fig)

# Display Individual Recency, Frequency, and Monetary
st.subheader('RFM Analysis')

# 1. Recency Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(rfm_df["recency"], bins=30, color=LIGHT_BLUE, kde=True, ax=ax)
ax.set_title("Recency Distribution", fontsize=20, color=NAVY_BLUE)
ax.set_xlabel("Recency (days)", fontsize=15, color=NAVY_BLUE)
ax.set_ylabel("Customer Count", fontsize=15, color=NAVY_BLUE)
st.pyplot(fig)

# 2. Frequency Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(rfm_df["frequency"], bins=30, color=LIGHT_BLUE, kde=True, ax=ax)
ax.set_title("Frequency Distribution", fontsize=20, color=NAVY_BLUE)
ax.set_xlabel("Number of Orders", fontsize=15, color=NAVY_BLUE)
ax.set_ylabel("Customer Count", fontsize=15, color=NAVY_BLUE)
st.pyplot(fig)

# 3. Monetary Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(rfm_df["monetary"], bins=30, color=LIGHT_BLUE, kde=True, ax=ax)
ax.set_title("Monetary Distribution", fontsize=20, color=NAVY_BLUE)
ax.set_xlabel("Monetary Value (Order Count)", fontsize=15, color=NAVY_BLUE)
ax.set_ylabel("Customer Count", fontsize=15, color=NAVY_BLUE)
st.pyplot(fig)

# NEW: Distribution of Canceled Orders Over the Last 3 Years
st.subheader('Distribution of Canceled Orders (Last 3 Years)')
canceled_orders_df = all_df[all_df["order_status"] == "canceled"]
canceled_orders_yearly = canceled_orders_df.groupby(canceled_orders_df["order_purchase_timestamp"].dt.year).order_id.nunique().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="order_purchase_timestamp", y="order_id", data=canceled_orders_yearly, palette='Blues_r', ax=ax)
ax.set_title("Number of Canceled Orders by Year", fontsize=20, color=NAVY_BLUE)
ax.set_xlabel("Year", fontsize=15, color=NAVY_BLUE)
ax.set_ylabel("Number of Canceled Orders", fontsize=15, color=NAVY_BLUE)
st.pyplot(fig)

# NEW: Top 10 Cities with Delivered Orders
st.subheader('Top 10 Cities by Delivered Orders')
top_10_cities_delivered = delivered_df.groupby("customer_city")["order_count"].sum().nlargest(10).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="order_count", y="customer_city", data=top_10_cities_delivered, palette='Blues_r', ax=ax)
ax.set_title("Top 10 Cities by Delivered Orders", fontsize=20, color=NAVY_BLUE)
ax.set_xlabel("Number of Orders", fontsize=15, color=NAVY_BLUE)
ax.set_ylabel("City", fontsize=15, color=NAVY_BLUE)
st.pyplot(fig)

# NEW: Comparison of New Customers by Month in 2017
st.subheader('Comparison of New Customers by Month in 2017')
new_customers_2017 = all_df[all_df["order_purchase_timestamp"].dt.year == 2017]
new_customers_monthly = new_customers_2017.groupby(new_customers_2017["order_purchase_timestamp"].dt.month)["customer_id"].nunique().reset_index()
new_customers_monthly.columns = ["Month", "New Customer Count"]

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="Month", y="New Customer Count", data=new_customers_monthly, palette='Blues_r', ax=ax)
ax.set_title("New Customers by Month in 2017", fontsize=20, color=NAVY_BLUE)
ax.set_xlabel("Month", fontsize=15, color=NAVY_BLUE)
ax.set_ylabel("New Customer Count", fontsize=15, color=NAVY_BLUE)
st.pyplot(fig)
