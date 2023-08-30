import numpy as np
import turtle
from scipy.interpolate import interp1d
import time
import numpy as np

def create_continuous_complex_function(arr, T):
    n = len(arr)
    x = np.linspace(0, T, n)  # Create x values for interpolation

    #Interpulate the complex function by interpolating its real and imaginary parts
    real_interpolated = interp1d(x, np.real(arr), kind='cubic')
    imag_interpolated = interp1d(x, np.imag(arr), kind='cubic')
    
    def f(x):
        if x < 0 or x > T:
            raise ValueError("Input value must be in the range [0, T]")
        
        return complex(real_interpolated(x), imag_interpolated(x))
    
    return f



def calculate_complex_coefficients(f, T, N):
    #Use the dst calculated by the fft to calculate the comeplx fourier series coofficients in O(NlogN) using numerical integration 
    #Generate N uniformly spaced samples for computing the DFT using the FFT
    t = np.linspace(0, T, N+1) 
    samples = [f(i) for i in t]
    #Devide by the number of samples to complete the numerical integration for finding the coofficients with a non-negative index
    pos_coefs = np.fft.fft(samples) / t.size 
    #The (-n)th complex fourier series coofficient of a complex function is equal to the complex conjugate of the (n)th complex fourier series coofficient of its conjugate function for every integer n
    #Use the fact above to calculate the negative indexed coofficients as previously described 
    neg_coefs = np.conjugate(np.fft.fft(np.conjugate(samples)))[1:] / t.size
    neg_coefs = np.flip(neg_coefs)
    #Unite the two arrays to an array of length 2*N+1
    return np.concatenate((neg_coefs, pos_coefs))



def series_exp_term(x, n, T):
        return np.exp(1j * 2 * np.pi * n * x / T)

def draw_lines_sum(coefs, temp_pen, dt, T):
    #This function uses the calculated complex fourier series to regenerate the user's drawing in T(s) with update intervals of length dt(s)
    #temp_pen drawn the line between each pair of consecutive rotating vectors with a decreasing magnitude of angular velocity

    #Start drawing from the middle
    temp_pen.penup()
    temp_pen.goto(0, 0)
    temp_pen.pendown()

    #Calculate the number of counter-clockwise rotating vectors
    pos_terms = int((len(coefs)-1)/2) 

    #Draw the first line (constant term)
    temp_pen.goto(coefs[pos_terms].real, coefs[pos_terms].imag)

    # Draw lines between consecutive points
    res = coefs[pos_terms]
    for npos in range(1, pos_terms+1):
        #Update each term in the fourier series according to the time that elapsed 
        coefs[pos_terms+npos] *= series_exp_term(dt,npos,T)
        coefs[pos_terms-npos] *= series_exp_term(dt,-npos,T)
        #Draw the relevant lines to the npos and -npos fourier terms 
        res+=coefs[pos_terms+npos]
        temp_pen.goto(res.real, res.imag)
        res+=coefs[pos_terms-npos]
        temp_pen.goto(res.real, res.imag)
    return res



def handle_drawn_points(points, drawing_duration=10, interval_length=0.001, clockwise_vectors=300, resolution_coef=100):
    #This function is the beating heart of the program and spcifies the different parameters for regenrating the user's drawing as a sum of harmonics
    #Note that the drawing of the user can be looked at as a complex function dependent on time
    #The fft assumes a periodic equally spaced data set, which makes drawing that are loops more accurate when generated

    #points; points accumulated by turtle to capture the user's drawing
    #drawing_duration; The time it takes to generate the entire drawing 
    #interval_length; The time interval between each update of the regenerated drawing 
    #clockwise_vectors; The number of clockwise vectors (similar to anti... ones) used to reconstruct the drawing 
    #resolution_coef; resultion_coef*len(points) is the number of samples in the input of the fft to generate the fourier series coofficients via numerical intergration

    T = drawing_duration # in seconds
    # Coefficients of the bounded Fourier series 
    pos_terms_for_calculation = resolution_coef*len(points)
    #Calculate and truncate the fourier coofficients array to the proper length  
    coefs = calculate_complex_coefficients(create_continuous_complex_function(points, T), T, pos_terms_for_calculation)
    coefs = coefs[pos_terms_for_calculation-clockwise_vectors:pos_terms_for_calculation+clockwise_vectors+1]
    turtle.delay(0)
    main_pen = turtle.Turtle()
    main_pen.color("blue")
    main_pen.hideturtle()
    main_pen.speed(0)  # Fastest drawing speed
    
    temp_pen = turtle.Turtle()
    temp_pen.hideturtle()
    temp_pen.speed(0)  # Fastest drawing speed

    # Use the computation above to visualize the image based on the coefficients and the Fourier Series formula
    drawing_time = 0 # in seconds
    dt = interval_length

    while(drawing_time<T):
        #Sleep and update the time passed 
        time.sleep(dt)
        drawing_time+=dt

        temp_pen.clear() #Clear the screen

        # Draw the line to the new point in the regenerated drawing 
        new_point = draw_lines_sum(coefs, temp_pen, drawing_time, T)
        main_pen.goto(new_point.real, new_point.imag) 
        
        if main_pen.isdown==False:
            main_pen.pendown()


def main():
    # Initialize the screen
    screen = turtle.Screen()
    screen.title("Drawing reconstruction with Turtle and The Fourier Series of a complex function")

    # Create a Turtle object
    pen = turtle.Turtle()
    pen.speed(0)  # Fastest drawing speed

    # Initialize the list to store complex numbers
    drawn_points = []

    # Function to handle mouse drag
    def drag(x, y):
        pen.ondrag(None)  # Disable event during this function
        pen.goto(x, y)
        drawn_points.append(complex(x, y))  # Save complex number x + y*i
        pen.ondrag(drag)  # Re-enable event after function

    # Set up the drag event
    pen.ondrag(drag)

    # Create a submit button
    submit_button = turtle.Turtle()
    submit_button.penup()
    submit_button.goto(0, -250)
    submit_button.write("Submit", align="center", font=("Arial", 14, "normal"))

    # Function to handle submit button click
    def submit_click(x, y):
        # Clear the screen
        pen.clear()
        time.sleep(0.1)

        handle_drawn_points(drawn_points)
        drawn_points.clear()  # Empty the array after handling

    # Bind the submit button to the click event
    submit_button.onclick(submit_click)

    # Run the Turtle graphics loop
    turtle.mainloop()

    # Close the Turtle graphics window on click
    screen.exitonclick()

if __name__ == "__main__":
    main()
