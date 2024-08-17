/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include "torch/torch.h"
#include "torch/script.h"


using namespace c74::min;


class hello_world : public object<hello_world>, public sample_operator<1, 0> {
public:
    MIN_DESCRIPTION	{"Post to the Max Console."};
    MIN_TAGS		{"utilities"};
    MIN_AUTHOR		{"Cycling '74"};
    MIN_RELATED		{"print, jit.print, dict.print"};

    inlet<>  input1	{ this, "(bang) post greeting to the max console" };
    inlet<>  input2  { this, "(signal) audio to measure / visualize" };
    outlet<> output	{ this, "(anything) output the message which is posted to the max console" };


    // define an optional argument for setting the message
    argument<symbol> greeting_arg { this, "greeting", "Initial value for the greeting attribute.",
        MIN_ARGUMENT_FUNCTION {
            cout << "setting greeting to " << arg << endl;
            greeting = arg;
        }
    };


    // the actual attribute for the message
    attribute<symbol> greeting { this, "greeting", "hello world",
        description {
            "Greeting to be posted. "
            "The greeting will be posted to the Max console when a bang is received."
        }
    };


    // respond to the bang message to do something
    message<> bang { this, "bang", "Post the greeting.",
        MIN_FUNCTION {
            symbol the_greeting = greeting;    // fetch the symbol itself from the attribute named greeting

            cout << the_greeting << endl;    // post to the max console
            output.send(the_greeting);       // send out our outlet
            return {};
        }
    };

    message<> testaa { this, "testaa", "testaa",
        MIN_FUNCTION {
            cout << args << endl;    // post to the max console
            return {};
        }
    };


    // post to max window == but only when the class is loaded the first time
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {
            cout << "hello world" << endl;
            return {};
        }
    };

    hello_world() {
        if (torch::hasCUDA()) {
            cout << "Has CUDA" <<endl;
        } else {
            cout << "Torch using CPU" << endl;
        }
        m_timer.delay(100);
    }
    timer<timer_options::defer_delivery> m_timer{ this,
        MIN_FUNCTION {
            output.send(m_unclipped_value);
            //cout << m_unclipped_value << endl;
            
            m_timer.delay(40);
            return {};
        }
    };

    void operator()(sample input) {
        m_unclipped_value = input;
        
    }

private:
    number m_unclipped_value{ 0.0 };

};


MIN_EXTERNAL(hello_world);
