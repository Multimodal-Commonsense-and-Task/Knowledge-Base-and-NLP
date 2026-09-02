# Action Primitives - Names and Descriptions

## List of Action Primitives

1. **OPEN([target_obj])**
   - Description: Open a [target_obj]

2. **CLOSE([target_obj])**
   - Description: Close a [target_obj]

3. **PLACE_ON_TOP([target_obj], [placement_obj])**
   - Description: Place the [target_obj] on top of [placement_obj]

4. **PLACE_INSIDE([target_obj], [placement_obj])**
   - Description: Place the [target_obj] inside [placement_obj]

5. **TOGGLE_ON([target_obj])**
   - Description: Toggle an [target_obj] on

6. **TOGGLE_OFF([target_obj])**
   - Description: Toggle an [target_obj] off

7. **WIPE([target_obj], [cleaning_tool])**
   - Description: Wipe the [target_obj] with the [cleaning_tool]

8. **CUT([target_obj], [cutting_tool])**
   - Description: Cut (slice or dice) the [target_obj] with the [cutting_tool]

9. **SOAK_UNDER([target_obj], [fluid_source])**
   - Description: Soak the [target_obj] with particles produced by the [fluid_source] (e.g., sink)

10. **SOAK_INSIDE([target_obj], [fluid_container])**
    - Description: Soak the [target_obj] with particles in the [fluid_container]

11. **FILL_WITH([target_obj], [fluid_source])**
    - Description: Fill the [target_obj] with particles produced by the [fluid_source] (e.g., sink)

12. **POUR_INTO([fluid_container], [target_obj])**
    - Description: Pour the particle in the [fluid_container] into the [target_obj] (usually a container)

13. **WAIT_FOR_COOKED([target_obj])**
    - Description: Wait for the cook process of the [target_obj] to final

14. **WAIT_FOR_WASHED([wash_machine])**
    - Description: Wait for the wash process of the [wash_machine] (e.g., dishwasher, washer)

15. **WAIT([target_obj])**
    - Description: Wait for the object to change, such as waiting for cooling down from heat or thawing from a frozen state

16. **WAIT_FOR_FROZEN([target_obj], [refrigerator_obj])**
    - Description: Wait for the target_obj inside refrigerator to frozen, such as waiting for quiche in electric_refrigerator to frozen

17. **SPREAD([liquid_container], [target_obj])**
    - Description: Spread some liquid from [liquid_container] onto [target_obj], make [target_obj] covered with these liquid particles

18. **DONE()**
    - Description: Indicate that the task has ended

---

## Parameter Count Summary

- **1 parameter actions**: OPEN, CLOSE, TOGGLE_ON, TOGGLE_OFF, WAIT, WAIT_FOR_COOKED, WAIT_FOR_WASHED
- **2 parameter actions**: PLACE_ON_TOP, PLACE_INSIDE, WIPE, CUT, SOAK_INSIDE, SOAK_UNDER, FILL_WITH, POUR_INTO, SPREAD, WAIT_FOR_FROZEN
- **0 parameters**: DONE

