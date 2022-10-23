_**ATR TRAILING STOPS EXIT**_

ATR Trailing Stops are primarily used to protect capital and lock in profits on individual trades but they can also be used, in conjunction with a trend filter, to signal entries.

_**ATR Trailing Stop Signals are used for exits:**_

- Exit your long position (sell) when price crosses below the ATR trailing stop line.
- Exit your short position (buy) when price crosses above the ATR trailing stop line.

_**ATR Trailing Stops Formula**_

_Trailing stops are normally calculated relative to closing price:_

- Calculate Average True Range ("ATR")
- Multiply ATR by your selected multiple — in our case 3 x ATR
- In an up-trend, subtract 3 x ATR from Closing Price and plot the result as the stop for the following day
- If price closes below the ATR stop, add 3 x ATR to Closing Price — to track a Short trade
- Otherwise, continue subtracting 3 x ATR for each subsequent day until price reverses below the ATR stop
- We have also built in a ratchet mechanism so that ATR stops cannot move lower during a Long trade nor rise during a Short trade.
