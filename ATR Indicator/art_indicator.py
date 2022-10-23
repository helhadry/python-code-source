import math
from datetime import datetime, timedelta

import dateutil
import numpy as np
import pandas as pd
from tqdm import tqdm

pd.options.mode.chained_assignment = None


class LoadContracts:
    """
    Class to load the cantracts details on which the calculation will be done
    """
    def __init__(self):
        self.path = path_to_contracts
        self.contracts = self.load_contracts()

    @staticmethod
    def string_to_datetime(x):
        return dateutil.parser.parse(x, dayfirst=True)

    def load_contracts(self):
        contracts = pd.read_csv(self.path)
        contracts["Date/Time"] = contracts["Date/Time"].map(lambda x: self.string_to_datetime(x))
        return contracts


class LoadMasterData:
    """
    Class to load master date and do the cleaning
    """

    COLUMNS = ["Ticker", "Date/Time", "Open", "High", "Low", "Close", "Volume", "Open Interest"]

    def __init__(self, ticker, start_date, end_date, time_frame):
        self.data_path = path_to_data_master
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.time_frame = time_frame

        self.master_data = pd.DataFrame()
        self.master_data_ticker = pd.DataFrame()

    @staticmethod
    def load_path(self):
        from google.colab import drive
        drive.mount('/content/drive')

    def load_data_from_csv(self):
        master_data = pd.read_csv(self.data_path)
        try:
            temp_dataframe = master_data.reset_index()
        except:
            temp_dataframe = master_data.reset_index(drop=True)
        self.master_data = temp_dataframe

    @staticmethod
    def string_to_datetime(x):
        return dateutil.parser.parse(x, dayfirst=True)

    def get_range_data(self):
        self.master_data["Date/Time"] = self.master_data["Date/Time"].map(lambda x: self.string_to_datetime(x))
        print("data size BEFORE selecting start and end dates: ", self.master_data.shape)
        self.master_data = self.master_data[(self.master_data["Date/Time"] >= self.start_date) & (self.master_data["Date/Time"] <= self.end_date)]
        print("data size AFTER selecting start and end dates: ", self.master_data.shape)
        self.master_data = self.master_data.sort_values(['Ticker', 'Date/Time'],
                                                        ascending=[False, True], ignore_index=True)

    def remove_extra_entries(self, remove_from_hour=23, remove_from_minute=30):
        print(f"Dataframe shape was {self.master_data.shape}")

        # function to choose whether entry is older then remove_from_hour:remove_from_minute time.
        def valid_datetime(x):
            input_date = x["Date/Time"]
            day_end = input_date.replace(hour=remove_from_hour, minute=remove_from_minute)
            return input_date < day_end

        self.master_data = self.master_data[self.master_data.apply(valid_datetime, axis=1)]
        print(f"Cleaned dataframe shape is {self.master_data.shape}")

    @staticmethod
    def calculate_one_entry(new_history: list):
        """ gets list of entries as an argument. transforms all of entries to one entry.
        """
        changed_entry = [
            new_history[0][0],  # name
            new_history[0][1],  # timestamp
            new_history[0][2],  # open_price
            max([x[3] for x in new_history]),  # highest price
            min([x[4] for x in new_history]),  # lowest price
            new_history[-1][5],  # close price
            sum([x[6] for x in new_history]),  # volume sum
            new_history[-1][7],  # open interest
        ]
        return changed_entry

    @staticmethod
    def check_date_new_format_multiplier(input_date, new_format, start_hour=9, start_minute=00):
        """ checks if input_date is start time multiplayer:
            if input_date == 9:15 + new_format * X -> returns True, Otherwise False.
        """
        day_start = input_date.replace(hour=start_hour, minute=start_minute)
        delta = max(input_date, day_start) - min(input_date, day_start)
        return (delta.seconds % (new_format * 60)) == 0

    @staticmethod
    def split_list_to_consecutive_lists(ticker_history_list):
        """ There can be some entries missing from history. That is why we need to make history consecutive.
        """
        result = []
        current_list = [ticker_history_list[0]]
        last_date = ticker_history_list[0][1]
        for i in range(1, len(ticker_history_list)):
            current_entry = ticker_history_list[i]
            current_date = current_entry[1]
            if (current_date - last_date).seconds == 60:
                current_list.append(current_entry)
            else:
                result.append(current_list)
                current_list = [current_entry]
            last_date = current_date

        result.append(current_list)
        return result

    def change_time_stamp(self, dataframe, new_format):
        # getting dataframe as python dictionary
        dataframe_list = dataframe.values.tolist()
        dataframe_dict = {}

        for item in dataframe_list:
            key = item[0]
            dataframe_dict[key] = dataframe_dict.get(key, []) + [item]

        # changing time format
        changed_dataframe_list = []
        for ticker, ticker_history_list in tqdm(dataframe_dict.items()):

            consecutive_history = self.split_list_to_consecutive_lists(ticker_history_list)
            for consecutive_list in consecutive_history:

                # correct to 9:15AM IST
                for begin_index in range(0, len(consecutive_list), 1):
                    begin_time = consecutive_list[begin_index][1]
                    if self.check_date_new_format_multiplier(begin_time, new_format):
                        break

                for start_index in range(begin_index, len(consecutive_list) - new_format + 1, new_format):
                    end_index = start_index + new_format
                    new_history = consecutive_list[start_index: end_index]

                    changed_entry = self.calculate_one_entry(new_history)
                    changed_dataframe_list.append(changed_entry)

        # build dataframe from changed history

        changed_dataframe = pd.DataFrame(changed_dataframe_list, columns=dataframe.columns)
        return changed_dataframe

    @staticmethod
    def calculate_bands(dfinp, period, multiplier):
        df = dfinp.copy()
        df['H - L'] = df['High'] - df['Low']
        df["CloseP"] = df["Close"].shift(1)
        df["|H - Cp|"] = abs(df['High'] - df["CloseP"])
        df["|L - Cp|"] = abs(df['Low'] - df["CloseP"])
        df["TR"] = df[['H - L', "|H - Cp|", "|L - Cp|"]].max(axis=1)
        ATR_list = [np.nan for i in range(period)]
        meani = round(df["TR"][1:period + 1].mean(), 2)
        ATR_list.append(meani)
        for tr in df['TR'][period + 1:]:
            meani = round(((meani * (period - 1) + tr) / period), 2)
            ATR_list.append(meani)
        df['ATR'] = ATR_list
        df['ATRlow'] = df['Close'] - multiplier * df['ATR']
        df['ATRhigh'] = df['Close'] + multiplier * df['ATR']
        df['lowerband'] = df['ATRlow'].rolling(period).max()
        df['upperband'] = df['ATRhigh'].rolling(period).min()
        df['lowerband'] = df['lowerband'].shift(1)
        df['upperband'] = df['upperband'].shift(1)
        del df['H - L'], df['CloseP'], df['|H - Cp|'], df['|L - Cp|']
        return df

    def run(self):
        self.load_data_from_csv()
        self.get_range_data()
        self.remove_extra_entries()

        self.master_data = self.master_data.drop(columns="index")
        if self.time_frame != 1:
            self.master_data = self.change_time_stamp(dataframe=self.master_data, new_format=self.time_frame)


class AtrTsl:
    """
    Class to calculate the ATR indicator
    """
    def __init__(self, master_data, master_data_ticker, ticker, contract, timeframe, date_time):
        self.master_data = master_data
        self.master_data_ticker = master_data_ticker
        self.ticker = ticker
        self.contract = contract
        self.timeframe = timeframe
        self.date_time = date_time
        self.master_data_ATR_TSL_output = pd.DataFrame()
        self.buy_time = None
        self.buy_price = None
        self.sell_time = None
        self.sell_price = None
        self.index_buy_price = None
        self.index_sell_price = None
        self.returns_contract = None
        self.returns_index = None
        self.date_is_valide = True

    def add_timeframe(self):
        self.date_time = self.date_time + timedelta(minutes=self.timeframe)

    @staticmethod
    def get_ATR_TSL(idf):
        master_data_ATR = idf.copy()
        upbl = master_data_ATR['upperband'].to_list()
        j = 0
        for i in upbl:
            if math.isnan(i):
                j += 1
            else:
                break
        if j:
            j -= 1

        lowl = master_data_ATR['ATRlow'].to_list()[j:]
        highl = master_data_ATR['ATRhigh'].to_list()[j:]
        lowbl = master_data_ATR['lowerband'].to_list()[j:]
        upbl = master_data_ATR['upperband'].to_list()[j:]
        closel = master_data_ATR['Close'].to_list()[j:]

        master_data_ATR['Close>ub'] = master_data_ATR['Close'] > master_data_ATR['upperband']
        master_data_ATR['Close>lb'] = master_data_ATR['Close'] > master_data_ATR['lowerband']
        master_data_ATR['Close<ub'] = master_data_ATR['Close'] < master_data_ATR['upperband']
        master_data_ATR['Close<lb'] = master_data_ATR['Close'] < master_data_ATR['lowerband']

        ATR_SL_list = [np.nan for i in range(j + 1)]
        ATR_Trail = [np.nan for i in range(j + 1)]
        initialize = True
        close_greater_condition = True
        count = 0
        for low, high, lb, ub, close in zip(lowl[1:], highl[1:], lowbl[1:], upbl[1:], closel[1:]):
            count += 1
            if initialize:
                if close_greater_condition:
                    tATRSL = min(lowl[count - 1], lb)
                    ATR_SL_list.append(tATRSL)
                else:
                    tATRSL = max(highl[count - 1], ub)
                    ATR_SL_list.append(tATRSL)

                initialize = False
            else:
                if close_greater_condition:
                    tATRSL = max(min(lowl[count - 1], lb), tATRSL)
                    ATR_SL_list.append(tATRSL)
                else:
                    tATRSL = min(max(highl[count - 1], ub), tATRSL)
                    ATR_SL_list.append(tATRSL)

            if close_greater_condition:
                trail = tATRSL > close
                ATR_Trail.append(trail)
                if trail:
                    initialize = True
                    close_greater_condition = False
            else:
                trail = tATRSL < close
                ATR_Trail.append(trail)
                if trail:
                    initialize = True
                    close_greater_condition = True

        master_data_ATR['ATR_SL'] = ATR_SL_list
        master_data_ATR['CLOSE>/< ATR TRAIL STOPS'] = ATR_Trail

        return master_data_ATR

    def get_result(self):
        if add_timeframe_to_contract:
            self.add_timeframe()

        df = self.master_data_ATR_TSL_output[['Ticker', 'Date/Time', 'Open', 'Close', 'ATR_SL']].copy()
        day_prices_index = df[df["Date/Time"].dt.date == self.date_time.date()]

        if self.date_time > day_prices_index.iloc[-1]['Date/Time']:
            self.date_is_valide = False
            return

        if self.contract.endswith("CE"):
            df["CLOSE>/< ATR TRAIL STOPS"] = np.where(df['Close'] < df["ATR_SL"], True, False)
            type_contract = "CE"
        elif self.contract.endswith("PE"):
            df["CLOSE>/< ATR TRAIL STOPS"] = np.where(df['Close'] > df["ATR_SL"], True, False)
            type_contract = "PE"

        df = df[df["Date/Time"] >= self.date_time]
        atr_stop = df[df["CLOSE>/< ATR TRAIL STOPS"] == True]

        first_row = df.iloc[0]
        self.buy_time = first_row["Date/Time"]

        self.index_buy_price = first_row["Open"]

        if len(atr_stop) == 0 or atr_stop.iloc[0]["Date/Time"].date() > self.date_time.date():
            self.sell_time = day_prices_index.iloc[-1]["Date/Time"]
            self.index_sell_price = day_prices_index.iloc[-1]["Close"]
        else:
            atr_stop_row = atr_stop.iloc[0]
            self.index_sell_price = atr_stop_row["Close"]
            self.sell_time = atr_stop_row["Date/Time"]

        master_data = self.master_data

        mask = (master_data["Date/Time"] >= self.buy_time) \
               & (master_data["Date/Time"] <= self.sell_time)\
               & (master_data["Ticker"].str.startswith(ticker)) \
               & (master_data["Ticker"].str.endswith(self.contract))

        master_data = master_data.loc[mask]

        self.buy_price = master_data.iloc[0]["Open"]
        self.sell_price = master_data.iloc[-1]["Close"]

        returns_contract = round(self.sell_price - self.buy_price, 1)

        if type_contract == "CE":
            returns_index = round(self.index_sell_price - self.index_buy_price, 1)
        else:
            returns_index = round(self.index_buy_price - self.index_sell_price, 1)

        self.returns_contract = returns_contract
        self.returns_index = returns_index

    def run(self):
        self.master_data_ATR_TSL_output = self.get_ATR_TSL(self.master_data_ticker)
        self.get_result()

    def get_atr_tsl_dataframe(self):
        return self.master_data_ATR_TSL_output


def save_results(ticker, timeframe, start_date, end_date, coincide_no_coindice,add_timeframe_to_contract,
                 period= None, multiplier= None):

    COLUMNS = ["Contract", "Buy time", "Buy price","Sell time", "Sell price", "Returns"]

    output_contract = pd.DataFrame(columns=COLUMNS)
    output_index = pd.DataFrame(columns=COLUMNS)

    contracts = LoadContracts().contracts
    loaddata = LoadMasterData(ticker, start_date, end_date, timeframe)
    loaddata.run()
    master_data = loaddata.master_data

    for index, row in contracts.iterrows():
        contract = row["Contract"]
        date_time = row["Date/Time"]

        if "Period" in row:
            period = row["Period"]
            multiplier = row["Multiplier"]

        master_data_ticker = master_data[master_data['Ticker'] == ticker].reset_index(drop=True)
        master_data_ticker = loaddata.calculate_bands(master_data_ticker, period, multiplier)

        # calculate the ART indicator
        atrTsl = AtrTsl(master_data, master_data_ticker, ticker, contract, timeframe, date_time)
        atrTsl.run()
        if atrTsl.date_is_valide:
            output_contract.loc[len(output_contract)] = [contract, atrTsl.buy_time, atrTsl.buy_price, atrTsl.sell_time,
                                       atrTsl.sell_price, atrTsl.returns_contract]

            output_index.loc[len(output_index)] = [contract, atrTsl.buy_time, atrTsl.index_buy_price, atrTsl.sell_time,
                                                   atrTsl.index_sell_price, atrTsl.returns_index]

    if coincide_no_coindice == "no coincide":
        first_contract = True
        for index, row in output_contract.iterrows():
            if first_contract:
                sell_time = row["Sell time"]
                first_contract = False
            else:
                if row["Buy time"] <= sell_time:
                    output_contract.loc[index, "Returns"] = 0
                else:
                    sell_time = row["Sell time"]

        first_index = True
        for index, row in output_index.iterrows():
            if first_index:
                sell_time = row["Sell time"]
                first_index = False
            else:
                if row["Buy time"] <= sell_time:
                    output_index.loc[index, "Returns"] = 0
                else:
                    sell_time = row["Sell time"]

    #save the contracts returns file
    output_contract.loc[len(output_contract)] = ['', '', '', '', 'Total returns', output_contract["Returns"].sum()]
    output_contract.to_csv(f"Contract returns_{coincide_no_coindice}_P1M2.csv", index=False)

    #save the index returns file
    output_index.loc[len(output_index)] = ['', '', '', '', 'Total returns', output_index["Returns"].sum()]
    output_index.to_csv(f"Index returns_{coincide_no_coindice}_P1M2.csv", index=False)

    atr_stl_df = atrTsl.get_atr_tsl_dataframe()
    atr_stl_df.to_excel("ATR_STL.xlsx")


######### INPUT DATA
ticker = "BANKNIFTY"
coincide_no_coindice = "nocoincide" #or "no coincide"
start_date = datetime(2022, 9, 12, 9, 15)
end_date = datetime(2022, 9, 14, 15, 30)
timeframe = 1
add_timeframe_to_contract = True # False to uncheck the adding of N min
# Please make sure to put the correct path
path_to_contracts = r'contract.csv'
path_to_data_master = r'master.csv'
######### END INPUT DATA

######## static period #####
static_period = True # put False to uncheck the static period
period = 2
multiplier = 2
##### END STATIC PERIOD #####

if static_period:
    if period is None or multiplier is None:
        raise Exception("Please input the period and multiplier !")

if timeframe in [None, ""]:
    raise Exception("Please input the timeframe !")

save_results(ticker, timeframe, start_date, end_date, coincide_no_coindice, add_timeframe_to_contract,
             period=period, multiplier=multiplier)

print("Contracts and index return files saved ...")