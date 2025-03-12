from control_service import ControlService

def main():
    try:
        print("Starting O-Robot Control Service...")
        control_service = ControlService()
        control_service.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()