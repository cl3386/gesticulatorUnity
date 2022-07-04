using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Python.Runtime;
// using System.Collections;
// using System.Collections.Generic;
using UnityEngine;
using System.IO;


public class Gesticulator : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        //gesticulator inference
        Main();

        //copy bvh file(generated at Project Folder) into Assets folder
        copyBvh();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    static void copyBvh(){

        File.Copy(@"C:/Users/CY/Desktop/desktop/pythonRuntime/temp.bvh", @"C:/Users/CY/Desktop/desktop/pythonRuntime/Assets/temp.bvh");

    }
    static void Main()


        {
            Runtime.PythonDLL = @"C:\Users\CY\AppData\Local\Programs\Python\Python38\python38.dll";

            var PYTHON_HOME = Environment.ExpandEnvironmentVariables(@"C:\Users\CY\AppData\Local\Programs\Python\Python38");

            // Python 홈 설정.
            PythonEngine.PythonHome = PYTHON_HOME;
            // 모듈 패키지 패스 설정.
            PythonEngine.PythonPath = string.Join(

                Path.PathSeparator.ToString(),
                new string[] {
                  PythonEngine.PythonPath,
                    // pip하면 설치되는 패키지 폴더.
                     Path.Combine(PYTHON_HOME, @"Lib\site-packages"),  
                    // 개인 패키지 폴더
                    @"C:\Users\CY\Desktop\desktop\pythonRuntime\Assets\pythonCode",
                    @"C:\Users\CY\Desktop\desktop\pythonRuntime\Assets\pythonCode\gesticulator",
                    @"C:\Users\CY\Desktop\desktop\pythonRuntime\Assets\pythonCode\gesticulator\gesticulator",
                    @"C:\Users\CY\Desktop\desktop\pythonRuntime\Assets\pythonCode\gesticulator\gesticulator\visualization"
                }
            );
            // Python 엔진 초기화
            PythonEngine.Initialize();

            // Global Interpreter Lock을 취득
            using (Py.GIL())
            {
            
            // 개인 패키지 폴더의 Demo/demo.py 읽기.   
                dynamic demo = Py.Import("demo.demo");   // It uses  PythonEngine.PythonPath    
                dynamic motionFromPython = demo.main();

            }    // using GIL( Py.GIL() )

            // python 환경을 종료한다.
            PythonEngine.Shutdown();

        }  

    }

