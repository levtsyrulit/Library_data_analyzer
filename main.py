import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LibraryDataAnalyser:
    def __init__(self):
        self.libraries_df = pd.read_csv("library_data/libraries.csv")
        self.checkouts_df = pd.read_csv("library_data/checkouts.csv")
        self.customers_df = pd.read_csv("library_data/customers.csv")
        self.books_df = pd.read_csv("library_data/books.csv")
        self.__prepare_data(self.libraries_df, self.checkouts_df, self.customers_df, self.books_df)

    def __prepare_data(self, libraries_df, checkouts_df, customers_df, books_df):
        libraries_df.drop_duplicates(inplace=True)
        checkouts_df.drop_duplicates(inplace=True)
        customers_df.drop_duplicates(inplace=True)
        books_df.drop_duplicates(inplace=True)

        checkouts_df['date_checkout'] = pd.to_datetime(checkouts_df['date_checkout'], format='%Y-%m-%d', errors='coerce')
        checkouts_df['date_returned'] = pd.to_datetime(checkouts_df['date_returned'], format='%Y-%m-%d', errors='coerce')
        checkouts_df.dropna(inplace=True)
        checkouts_df["time_between"] = checkouts_df["date_returned"] - checkouts_df["date_checkout"]
        checkouts_df["late_return"] = checkouts_df["time_between"] > pd.Timedelta(days=28)

        customers_df.rename(columns={"id": "patron_id"}, inplace=True)

    def calculate_rate(self):
        self.checkouts_df["time_between"] = self.checkouts_df["date_returned"] - self.checkouts_df["date_checkout"]
        self.checkouts_df["late_return"] = self.checkouts_df["time_between"] > pd.Timedelta(days=28)

        num_late_returns = self.checkouts_df["late_return"].sum()
        num_total_returns = self.checkouts_df["late_return"].count()
        late_return_rate = num_late_returns / num_total_returns
        print(f"Number of late returns: {num_late_returns}")
        print(f"Total number of returns: {num_total_returns}")
        print(f"Late return rate: {late_return_rate:.2%}")

    def gender_rate(self):
        merged_df = pd.merge(self.checkouts_df, self.customers_df, on="patron_id")
        merged_df["gender"] = merged_df["gender"].str.strip().str.lower()
        merged_df = merged_df[(merged_df["gender"] == "male") | (merged_df["gender"] == "female")]

        sns.countplot(x="gender", hue="late_return", data=merged_df)
        plt.title("Late Returns by Gender")
        plt.show()
        male_return_rate = merged_df.loc[merged_df["gender"] == "male", "late_return"].mean()
        female_return_rate = merged_df.loc[merged_df["gender"] == "female", "late_return"].mean()

        print(f"Late return rate for male patrons: {male_return_rate:.2%}")
        print(f"Late return rate for female patrons: {female_return_rate:.2%}")

    def education_rate(self):
        merged_df = pd.merge(self.checkouts_df, self.customers_df, on="patron_id")
        merged_df["education"] = merged_df["education"].str.strip().str.lower().replace('\s+', ' ', regex=True)
        print(merged_df["education"].unique())

        sns.countplot(x="education", hue="late_return", data=merged_df)
        plt.title("Late Returns by Education Level")
        plt.show()

        for edu_level in merged_df["education"].unique():
            if pd.isnull(edu_level):
                continue  # Skip NaN values
            edu_level = edu_level.strip()  # Remove leading/trailing whitespace
            late_return_rate = merged_df.loc[merged_df["education"] == edu_level, "late_return"].mean()
            print(f"Late return rate for {edu_level} patrons: {late_return_rate:.2%}")

    def occupation_rate(self):
        merged_df = pd.merge(self.customers_df, self.checkouts_df, on='patron_id')
        # Filter out rows with missing checkout or return dates
        merged_df.dropna(subset=['date_checkout', 'date_returned'], inplace=True)
        #Clean up the occupation names
        merged_df['occupation'] = merged_df['occupation'].apply(lambda x: ' '.join(x.lower().split()) if isinstance(x, str) else None)
        #Remove rows with missing occupation values
        merged_df.dropna(subset=['occupation'], inplace=True)
        # Calculate the number of days late for each checkout (clipped to 0 and 28 days)
        merged_df['days_late'] = (pd.to_datetime(merged_df['date_returned']) - pd.to_datetime(merged_df['date_checkout']) - pd.Timedelta(days=28)).dt.days.clip(lower=0, upper=28)
        # Filter for checkouts that are more than 28 days late
        merged_df = merged_df[merged_df['days_late'] > 0]

        # Group by occupation and calculate the total number of checkouts for each group
        occupation_count = merged_df.groupby('occupation')['days_late'].count()
        # Group by occupation and calculate the number of late checkouts for each group
        occupation_late_count = merged_df.groupby('occupation')['days_late'].sum()

        occupation_late_rate = occupation_late_count / occupation_count

        plt.bar(occupation_late_rate.index, occupation_late_rate.values)
        plt.xticks(rotation=45)

        plt.title('Rate of Late Returns by Occupation')
        plt.xlabel('Occupation')
        plt.ylabel('Ratio')
        plt.show()
        for occupation in occupation_late_rate.index:
            print(f"{occupation.title()}: {occupation_late_rate.loc[occupation]:.2f}%")

    def library_rate(self):
        merged_df = pd.merge(self.customers_df, self.checkouts_df, on='patron_id')
        self.libraries_df["street_address"] = self.libraries_df["street_address"].str.strip().str.lower().replace('\s+', ' ',
                                                                                                        regex=True)
        library_ids = merged_df["library_id"].unique()
        names = []
        late_return_lib = []
        for library_id in library_ids:
            name = self.libraries_df[self.libraries_df["id"] == library_id]["street_address"].values[0]
            library_df = merged_df[merged_df["library_id"] == library_id]

            num_late_returns = library_df["late_return"].sum()
            num_total_returns = library_df["late_return"].count()
            late_return_rate = num_late_returns / num_total_returns

            print(f"{name}: {late_return_rate:.2%}")
            names.append(name)
            late_return_lib.append(late_return_rate)

        plt.bar(names, late_return_lib)
        plt.xticks(rotation=45)

        plt.title('Number of Late Returns by library')
        plt.xlabel('Library')
        plt.ylabel('Ratio')
        plt.show()

    def city_rate(self):
        merged_df = pd.merge(self.customers_df, self.checkouts_df, on="patron_id")

        #Remove rows with missing or null city values
        merged_df = merged_df.dropna(subset=["city"])
        #Standardize city names by removing extra spaces and converting to lowercase
        merged_df["city"] = merged_df["city"].str.strip().str.lower().str.replace(r'\s+', ' ')

        city_groups = merged_df.groupby("city")
        cities = []
        late_return_rate_list = []
        for city, city_df in city_groups:
            num_late_returns = city_df["late_return"].sum()
            num_total_returns = city_df["late_return"].count()
            late_return_rate = num_late_returns / num_total_returns
            print(f"Late return rate for {city}: {late_return_rate:.2%}")
            cities.append(city)
            late_return_rate_list.append(late_return_rate)

        plt.bar(cities, late_return_rate_list)
        plt.xticks(rotation=45)

        plt.title('Number of Late Returns by city')
        plt.xlabel('City')
        plt.ylabel('Ratio')
        plt.show()

obj = LibraryDataAnalyser()
#obj.calculate_rate()
#obj.gender_rate()
#obj.education_rate()
#obj.occupation_rate()
#obj.library_rate()
#obj.city_rate()


