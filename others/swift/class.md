```swift
class Car {
    var year: Int
    var make: String
    var color: String
    
    init(year: Int, make: String, color: String) {
        self.year = year 
        self.make = make 
        self.color = color
    }
}

var myCar = Car(year: 2022, make: "Porsche", color: "Grey")
var stolenCar = myCar // classes are reference types
stolenCar.color = "green"
print(myCar.color)

struct Car {
    var year: Int
    var make: String
    var color: String
}

var myCar = Car(year: 2022, make: "Porsche", color: "Grey")
var stolenCar = myCar // struct are value types
stolenCar.color = "green"
print(myCar.color) // "Grey"
```